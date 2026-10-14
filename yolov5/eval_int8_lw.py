"""
eval_int8_lw.py -- mAP của model INT8 ĐÃ DEPLOY (extracted_*), đo qua CHÍNH pipeline
inference_cpu.py (bit-exact với board — board đã chứng minh --verify max=0 mọi lớp).
Preprocessing = resize thẳng 416 (đúng như deploy); GT normalized khớp trực tiếp.
mAP tính bằng hạ tầng chuẩn yolov5 (val.process_batch + ap_per_class, IoU 0.5:0.95).

Tự-kiểm-chứng: chạy 2 model (lw50 non-uniform + uniform baseline) — INT8 mAP phải
land gần FP32 (lw50 FP32=0.8383, uniform FP32=0.7976) để tin số.

Usage: SPARSE_EXTRACTED=runs/quant/extracted_lw50 python eval_int8_lw.py [--n 250]
"""
import os, sys, argparse, glob, time
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from val import process_batch
from utils.metrics import ap_per_class

IMG = 416


def to_xyxy_px(lbl):                       # yolo norm xywh -> 416px xyxy
    c, x, y, w, h = lbl
    return [c, (x-w/2)*IMG, (y-h/2)*IMG, (x+w/2)*IMG, (y+h/2)*IMG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()
    os.environ.setdefault("CONF", str(args.conf))    # inference_cpu đọc env CONF cho decode
    import importlib
    import inference_cpu as IC
    importlib.reload(IC)                              # nạp lại với CONF thấp
    from PIL import Image

    print(f"[*] SPARSE_EXTRACTED={IC.E}  conf={args.conf}")
    meta, req, weights, bias = IC.load_model()
    conv, osc, isc = IC.make_runner(meta, req, weights, bias, verify=False)

    vdir = Path("../datasets/pklot/valid/images")
    imgs = sorted(glob.glob(str(vdir / "*.jpg")))[:args.n]
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []
    t0 = time.time()
    for ii, ip in enumerate(imgs):
        im = Image.open(ip).convert("RGB").resize((IMG, IMG))
        arr = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        x = np.round(arr / IC.INPUT_SCALE).clip(-128, 127).astype(np.int8)
        outs = IC.run_graph(x, conv, osc, isc)
        dets = IC.decode(outs, conf=args.conf, nms=IC.NMS_THRESH)  # [x1,y1,x2,y2,conf,cls]

        lp = ip.replace("images", "labels").replace(".jpg", ".txt")
        gt = []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    gt.append(to_xyxy_px([int(p[0])] + [float(v) for v in p[1:]]))
        gt = np.array(gt, dtype=np.float32).reshape(-1, 5)

        if len(dets) == 0:
            if len(gt):
                stats.append((np.zeros((0, 10), bool), np.array([]), np.array([]), gt[:, 0]))
            continue
        d = np.array(dets, dtype=np.float32)          # [N,6]
        d = d[np.argsort(-d[:, 4])]
        if len(gt) == 0:
            stats.append((np.zeros((len(d), 10), bool), d[:, 4], d[:, 5], np.array([])))
            continue
        correct = process_batch(torch.tensor(d), torch.tensor(gt), iouv).cpu().numpy()
        stats.append((correct, d[:, 4], d[:, 5], gt[:, 0]))
        if (ii + 1) % 50 == 0:
            print(f"  {ii+1}/{len(imgs)}  ({(time.time()-t0):.0f}s)", flush=True)

    st = [np.concatenate(v, 0) for v in zip(*stats)]
    tp, fp, p, r, f1, ap, cls = ap_per_class(*st, names={0: "empty", 1: "occ"})
    map50 = ap[:, 0].mean()
    map5095 = ap.mean()
    print(f"\n=== INT8 mAP ({len(imgs)} val imgs, conf={args.conf}) ===")
    print(f"  mAP50    = {map50:.4f}")
    print(f"  mAP50-95 = {map5095:.4f}")
    print(f"  P={p.mean():.3f} R={r.mean():.3f}")


if __name__ == "__main__":
    main()
