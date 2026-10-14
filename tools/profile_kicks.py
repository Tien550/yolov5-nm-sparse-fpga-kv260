#!/usr/bin/env python3
"""
profile_kicks.py — Analytic per-layer time breakdown of the FPGA inference.

Time on the FPGA is dominated by #kicks x per-kick cost (measured 0.224 ms/kick,
host-bound). This script traces the run_graph shapes (no weights, no board) and
computes the exact kick count each layer issues, using the SAME formulas the host
orchestrator (fpga_conv3x3.py) uses:

  conv3x3 / conv1x1 :  ceil(Cout/4) * (1 prime + H * ceil(W/16) * ceil(Cin/16))
  dwconv            :  ceil(C/4)    * (1 prime + H * ceil(W/16))
  fp32 {42,49,56,62}:  0 kicks (runs on CPU as float)

H,W are the *input* spatial dims (the orchestrator computes at full stride-1
resolution then subsamples for stride-2), so stride-2 layers cost 4x their output.

Output: per-layer kicks + est. time, and rollups by kind / by stage, so we can see
which part to target for real-time (host opt vs on-chip orchestration).
"""
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import inference_cpu as C

PER_KICK_MS = 0.224     # measured reliable per-kick host time (UIO+DMA bound)
WOUT = 16; CIN_CHUNK = 16

meta = json.load(open(f"{C.E}/metadata.json"))
req  = json.load(open(f"{C.E}/requant_params.json"))
osc  = lambda lid: meta[f"layer_{lid:02d}"]["output_scale"]
isc  = lambda lid: meta[f"layer_{lid:02d}"]["input_scale"]

records = []   # dict per conv layer

def k_conv(Cin, Cout, H, W):
    nch  = math.ceil(Cin / CIN_CHUNK)
    n_cb = math.ceil(W / WOUT)
    n_ocb = math.ceil(Cout / 4)
    return n_ocb * (1 + H * n_cb * nch)

def k_dw(Cch, H, W):
    n_cb = math.ceil(W / WOUT)
    n_ocb = math.ceil(Cch / 4)
    return n_ocb * (1 + H * n_cb)

def conv(lid, x, relu=True):
    k = f"layer_{lid:02d}"; m = meta[k]; ws = m["weight_shape"]
    Cout, Cin, kk = ws[0], ws[1], ws[2]
    stride = m["stride"][0]; pad = m["padding"][0]; is_dw = bool(m["is_depthwise"])
    _, H, W = x.shape
    Hout = (H + 2*pad - kk)//stride + 1
    Wout = (W + 2*pad - kk)//stride + 1
    # #1: conv3x3 stride-2 now kicks only the kept rows (Hout) instead of all H rows.
    # cols still full width (n_cb=ceil(W/16)); old = pre-#1 (all H rows).
    if k not in req:
        kind = "fp32_cpu"; nk = nk_old = 0
    elif is_dw:
        kind = "dwconv"; nk = k_dw(Cout, H, W); nk_old = nk          # dw all stride1
    elif kk == 1:
        kind = "conv1x1"; nk = k_conv(Cin, Cout, H, W); nk_old = nk  # 1x1 all stride1
    else:
        kind = "conv3x3"; nk = k_conv(Cin, Cout, Hout, W); nk_old = k_conv(Cin, Cout, H, W)
    records.append(dict(lid=lid, kind=kind, Cin=Cin, Cout=Cout, k=kk,
                        H=H, W=W, stride=stride, kicks=nk, kicks_old=nk_old))
    dt = np.float32 if k not in req else np.int8
    return np.zeros((Cout, Hout, Wout), dt)

# trace the real graph (shapes only)
shp = C.json.load(open(f"{C.GOLD}/layer_shapes.json"))["layer_00"]["in"]
img = np.zeros(shp, np.int8)
C.run_graph(img, conv, osc, isc)

records.sort(key=lambda r: r["lid"])
total = sum(r["kicks"] for r in records)

print(f"{'L':>3} {'kind':9} {'Cin':>4} {'Cout':>4} {'HxW':>9} {'st':>2} "
      f"{'kicks':>8} {'ms':>8} {'%':>5}")
print("-"*64)
for r in records:
    ms = r["kicks"] * PER_KICK_MS
    pct = 100*r["kicks"]/total if total else 0
    print(f"{r['lid']:>3} {r['kind']:9} {r['Cin']:>4} {r['Cout']:>4} "
          f"{r['H']:>3}x{r['W']:<3} {r['stride']:>2} {r['kicks']:>8} "
          f"{ms:>8.0f} {pct:>5.1f}")

total_old = sum(r["kicks_old"] for r in records)
print("-"*64)
print(f"TOTAL (post-#1) kicks={total:,}  est={total*PER_KICK_MS/1000:.1f}s "
      f"({total*PER_KICK_MS/1000/60:.1f} min) @ {PER_KICK_MS}ms/kick")
print(f"TOTAL (pre-#1)  kicks={total_old:,}  est={total_old*PER_KICK_MS/1000/60:.1f} min")
print(f"#1 STRIDE-2 ROW-SKIP SAVES {total_old-total:,} kicks "
      f"({100*(total_old-total)/total_old:.1f}%, "
      f"{(total_old-total)*PER_KICK_MS/1000/60:.1f} min)\n")

# rollup by kind
print("== by kind ==")
for kind in ["conv1x1","conv3x3","dwconv","fp32_cpu"]:
    sel=[r for r in records if r["kind"]==kind]
    kk=sum(r["kicks"] for r in sel)
    print(f"  {kind:9} layers={len(sel):>2} kicks={kk:>9,} "
          f"({100*kk/total:>5.1f}%)  est={kk*PER_KICK_MS/1000/60:>5.1f} min")

# rollup by stage (lid ranges)
print("\n== by stage ==")
stages=[("backbone C0-C29",range(0,30)),("SPPF 30-31",range(30,32)),
        ("head/neck 32-59",range(32,60)),("detect 60-62",range(60,63))]
for name,rng in stages:
    sel=[r for r in records if r["lid"] in rng]
    kk=sum(r["kicks"] for r in sel)
    print(f"  {name:18} kicks={kk:>9,} ({100*kk/total:>5.1f}%) "
          f"est={kk*PER_KICK_MS/1000/60:>5.1f} min")

# top-10 heaviest layers
print("\n== top-10 heaviest layers ==")
for r in sorted(records,key=lambda r:-r["kicks"])[:10]:
    print(f"  L{r['lid']:>2} {r['kind']:9} Cin={r['Cin']:>4} Cout={r['Cout']:>4} "
          f"{r['H']}x{r['W']} st{r['stride']}  {r['kicks']:>8,} kicks "
          f"({100*r['kicks']/total:>4.1f}%)")
