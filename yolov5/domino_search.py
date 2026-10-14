"""
domino_search.py -- DominoSearch (NeurIPS'21, Sun et al.) cho YOLOv5s-Lite conv1x1 N:16.
Bám Algorithm 3.1 với 2 điều chỉnh khai báo trước:
 (1) Cấu trúc CỘT-CHIA-SẺ: idx top-N per M-block CHUNG cho mọi oc (engine i_spec wt_rd_idx)
     → "group" của paper = M-block CỘT; phần tử = column-score s_c = mean_oc|w[:,c]|.
 (2) Complexity = mô hình CYCLE thật của engine (paper §3.1: FLOPs thay được bằng latency):
     cyc_i(N) = (HW/16)·(Cin/16)·ceil(Cout/48)·(4N+7)   [W11 4cyc/entry + 7 overhead/block]
Thành phần giữ nguyên paper: group threshold = mean của M/2 score nhỏ nhất; supermajority
vote (vr); penalty định kỳ trên KEPT weights (eq.4, STE dense-update); η_i = β1·c_i + β2·r_i
(c=complexity chuẩn hóa eq.6, r=ERK redundancy eq.7); domino descent tới khi Sp_c ≤ Sp_t.
Pool N = {1,2,3,4,6,8,16} (HW nhận n_cfg 1..16 — paper dùng power-of-2 chỉ là lựa chọn).
Usage: venv python domino_search.py [--budget 1.0] [--epochs 4] [--vr 0.7]
  --budget: Sp_t = budget × cycles(uniform N=3)  (1.0 = equal-complexity, paper Table 1)
Output: runs/sens/layerwise_nm.json  {layer: N}  + log domino descent.
"""
import torch, sys, warnings, argparse, math, json, yaml
import numpy as np
import torch.nn as nn
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr, init_seeds
from utils.loss import ComputeLoss

DATA = "data/pklot.yaml"; IMGSZ = 416; M = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
POOL = [16, 8, 6, 4, 3, 2, 1]          # descending (domino đi xuống)
OUTD = Path("runs/sens"); OUTD.mkdir(parents=True, exist_ok=True)


def col_scores(w):                      # w [Cout,Cin,1,1] -> [K,M] column scores
    Cout, Cin = w.shape[0], w.shape[1]
    return w.detach().abs().reshape(Cout, Cin).mean(0).reshape(Cin // M, M)


def group_thresholds(sc):               # paper §3.2: mean của M/2 score nhỏ nhất mỗi group
    smallest, _ = sc.topk(M // 2, dim=1, largest=False)
    return smallest.mean(1)             # [K]


def vote_N(sc, vr):
    """N_g[k] = #cột vượt threshold; layer-N = vr-quantile (supermajority relaxation)."""
    T = group_thresholds(sc)
    Ng = (sc > T.unsqueeze(1)).sum(1)   # [K]
    return int(np.quantile(Ng.cpu().numpy(), vr))


def snap_pool(n):                       # snap XUỐNG pool value gần nhất (≤ n); tối thiểu 1
    for p in POOL:
        if p <= n: return p
    return 1


def col_mask(w, sc):
    """mask cột theo threshold hiện tại (eq.3, dynamic — KHÔNG ép đúng-N khi search)."""
    T = group_thresholds(sc)
    keep = (sc > T.unsqueeze(1))        # [K,M] bool
    m = keep.reshape(1, -1, 1, 1).float().to(w.device)
    return m.expand_as(w).contiguous()


def cycles(hw, cin, cout, N):           # mô hình cycle engine (W11)
    return (hw / 16) * (cin / 16) * math.ceil(cout / 48) * (4 * N + 7)


def erk_sparsity(layers, target_overall):
    """ERK (RigL): density_i ∝ (Cin+Cout+kh+kw)/(Cin·Cout·kh·kw); scale để đạt target."""
    raw = {n: (i["cin"] + i["cout"] + 2) / (i["cin"] * i["cout"]) for n, i in layers.items()}
    tot = sum(layers[n]["cin"] * layers[n]["cout"] for n in layers)
    lo, hi = 0.0, 1e9
    for _ in range(60):                 # tìm hệ số scale bằng bisection
        mid = (lo + hi) / 2
        dens = {n: min(1.0, mid * raw[n]) for n in raw}
        kept = sum(dens[n] * layers[n]["cin"] * layers[n]["cout"] for n in layers)
        if kept / tot > 1 - target_overall: hi = mid
        else: lo = mid
    return {n: 1 - min(1.0, lo * raw[n]) for n in raw}   # e_i = sparsity ERK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=4)     # search ngắn; dừng sớm khi đạt Sp_t
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)   # 0.005 → vote sập 1 nhịp
    ap.add_argument("--vr", type=float, default=0.7)
    ap.add_argument("--kp", type=int, default=100)       # chu kỳ penalty (eq.5)
    ap.add_argument("--kc", type=int, default=100)       # chu kỳ check/vote (Alg 3.1 line 6)
    ap.add_argument("--b1", type=float, default=0.8)     # FLOPs-constraint: β1=0.8 (paper §3.4)
    ap.add_argument("--b2", type=float, default=0.2)
    ap.add_argument("--greedy", action="store_true",
                    help="η-ordered greedy descent (không train trong search — penalty L2 đều "
                         "là scale-invariant với vote nên descent train-based không tiến)")
    args = ap.parse_args(); init_seeds(0)

    ck = torch.load("runs/sparse/dense_parent.pt", map_location="cpu", weights_only=False)
    model = ck["model"].float().to(DEV)
    print(f"dense_parent: ep{ck.get('epoch')} mAP50={ck.get('mAP50'):.4f}")
    dd = check_dataset(DATA)
    if not hasattr(model, "names"): model.names = dd["names"]
    gs = max(int(model.stride.max()), 32)

    # ── thu thập layer conv1x1 + spatial size (chạy 1 forward hook lấy HW) ──
    convs = {}                          # name -> dict(mod, cin, cout, hw)
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1) and mod.weight.shape[1] % M == 0:
            convs[name] = {"mod": mod, "cin": mod.weight.shape[1], "cout": mod.weight.shape[0]}
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, n=name: convs[n].__setitem__("hw", o.shape[2] * o.shape[3])))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, IMGSZ, IMGSZ, device=DEV))
    for h in hooks: h.remove()
    print(f"conv1x1: {len(convs)} layers")

    # ── budget: Sp_t = budget × cycles(uniform N=3) ──
    cyc3 = sum(cycles(i["hw"], i["cin"], i["cout"], 3) for i in convs.values())
    Spt = args.budget * cyc3
    print(f"cycles(uniform N=3) = {cyc3:,.0f}  →  Sp_t = {Spt:,.0f} (budget {args.budget})")

    # ── ERK guidance (target overall = sparsity của uniform N=3 = 81.25%) ──
    erk = erk_sparsity(convs, 1 - 3 / 16)

    # ── init N_v từ dense weights (Alg 3.1 line 7-10 tại t=0) ──
    Nv = {}
    for n, i in convs.items():
        Nv[n] = snap_pool(vote_N(col_scores(i["mod"].weight), args.vr))
    def spc(): return sum(cycles(i["hw"], i["cin"], i["cout"], Nv[n]) for n, i in convs.items())
    print(f"init: Sp_c = {spc():,.0f}")

    hyp = yaml.safe_load(open("data/hyps/hyp.scratch-low.yaml"))
    for k, v in dict(box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, fl_gamma=0.0,
                     anchor_t=4.0, label_smoothing=0.0).items():
        hyp.setdefault(k, v)
    model.hyp = hyp; model.nc = dd["nc"]; model.gr = 1.0
    tl, _ = create_dataloader(dd["train"], imgsz=IMGSZ, batch_size=args.bs, stride=gs,
                              hyp=hyp, augment=True, cache=False, pad=0.0, rect=False,
                              workers=0, prefix=colorstr("train: "), shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    compute_loss = ComputeLoss(model)

    masks = {n: col_mask(i["mod"].weight, col_scores(i["mod"].weight)) for n, i in convs.items()}

    def eta(n):                          # η_i = β1·c_i + β2·r_i (eq.6-8)
        i = convs[n]; s = Nv[n] / M
        c_raw = {m: cycles(convs[m]["hw"], convs[m]["cin"], convs[m]["cout"], Nv[m])
                 for m in convs}
        c = c_raw[n] / max(c_raw.values())
        r_raw = {m: erk[m] - (1 - Nv[m] / M) for m in convs}
        mx = max(abs(v) for v in r_raw.values()) or 1.0
        r = r_raw[n] / mx
        return args.b1 * c + args.b2 * r

    if args.greedy:
        # ── GREEDY η-DESCENT: mỗi bước, lớp có η lớn nhất (trong các lớp còn hạ được)
        #    rơi đúng 1 nấc pool, tới khi Sp_c ≤ Sp_t. η = β1·c + β2·r y nguyên paper. ──
        print("=== GREEDY η-DESCENT (không train) ===")
        FLOOR = 2                              # không ép lớp nào xuống N=1 (bảo toàn mAP)
        steps = 0
        while spc() > Spt:
            cand = [(eta(n), n) for n in convs if Nv[n] > FLOOR]
            if not cand:
                print("!! hết lớp hạ được mà chưa đạt budget"); break
            _, pick = max(cand)
            Nv[pick] = POOL[min(POOL.index(Nv[pick]) + 1, len(POOL) - 1)]
            steps += 1
            if steps % 10 == 0:
                print(f"  step {steps}: Sp_c={spc():,.0f} ({spc()/cyc3:.2f}×N3)")
        print(f"  ★ ĐẠT budget sau {steps} bước: Sp_c={spc():,.0f} ≤ {Spt:,.0f}")
        scheme = {n: Nv[n] for n in sorted(Nv)}
        out = OUTD / "layerwise_nm.json"
        out.write_text(json.dumps({"scheme": scheme, "budget": args.budget,
                                   "cyc_uniform3": cyc3, "cyc_scheme": spc()}, indent=1))
        hist = {}
        for n in Nv: hist[Nv[n]] = hist.get(Nv[n], 0) + 1
        print("\n=== SCHEME ===")
        for n, N in scheme.items():
            i = convs[n]
            print(f"  {n:38s} Cin={i['cin']:4d} HW={i['hw']:5d} -> N={N}")
        print(f"N-hist: {dict(sorted(hist.items()))}  cycles = {spc()/cyc3:.2f} × uniform-N3")
        print(f"→ {out}")
        return

    print("=== DOMINO DESCENT ===")
    t = 0; done = False
    for ep in range(args.epochs):
        if done: break
        model.train()
        for imgs, targets, _, _ in tl:
            imgs = imgs.to(DEV).float() / 255; targets = targets.to(DEV)
            # forward/backward tại w̃ = w⊙m; update áp lên FULL w (STE, eq.4)
            stash = {n: convs[n]["mod"].weight.data.clone() for n in convs}
            for n in convs: convs[n]["mod"].weight.data.mul_(masks[n])
            loss, _ = compute_loss(model(imgs), targets)
            opt.zero_grad(); loss.backward()
            for n in convs: convs[n]["mod"].weight.data.copy_(stash[n])
            if t % args.kp == 0:         # penalty định kỳ trên KEPT weights (eq.4-5)
                for n in convs:
                    w = convs[n]["mod"].weight
                    if w.grad is None:      # layer ngoài graph/frozen — SGD cũng bỏ qua
                        continue
                    w.grad.add_(eta(n) * (w.data * masks[n]))
            opt.step()
            t += 1
            if t % args.kc == 0:         # Alg 3.1 line 6-16
                for n, i in convs.items():
                    sc = col_scores(i["mod"].weight)
                    nv = snap_pool(vote_N(sc, args.vr))
                    if nv < Nv[n]:
                        # DOMINO THẬT: rơi tối đa 1 nấc pool mỗi chu kỳ (chống sập 1 nhịp
                        # 4.37M→1.26M như run đầu — thứ tự rơi giữa các lớp = tín hiệu redundancy)
                        pi = POOL.index(Nv[n])
                        Nv[n] = POOL[min(pi + 1, len(POOL) - 1)]
                    masks[n] = col_mask(i["mod"].weight, sc)
                cur = spc()
                if t % (args.kc * 5) == 0:
                    hist = {}
                    for n in Nv: hist[Nv[n]] = hist.get(Nv[n], 0) + 1
                    print(f"  t={t} loss={loss.item():.3f} Sp_c={cur:,.0f} "
                          f"({cur/cyc3:.2f}×N3) N-hist={dict(sorted(hist.items()))}", flush=True)
                if cur <= Spt:
                    print(f"  ★ ĐẠT budget tại t={t}: Sp_c={cur:,.0f} ≤ Sp_t={Spt:,.0f}")
                    done = True; break

    scheme = {n: Nv[n] for n in sorted(Nv)}
    out = OUTD / "layerwise_nm.json"
    out.write_text(json.dumps({"scheme": scheme, "budget": args.budget,
                               "cyc_uniform3": cyc3, "cyc_scheme": spc()}, indent=1))
    hist = {}
    for n in Nv: hist[Nv[n]] = hist.get(Nv[n], 0) + 1
    print("\n=== SCHEME ===")
    for n, N in scheme.items():
        i = convs[n]
        print(f"  {n:38s} Cin={i['cin']:4d} HW={i['hw']:5d} -> N={N}")
    print(f"N-hist: {dict(sorted(hist.items()))}  cycles = {spc()/cyc3:.2f} × uniform-N3")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
