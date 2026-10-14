"""Full sanity check: metadata, golden shapes, cnn_pkg.sv, encode/decode."""
import json, os, sys, struct, numpy as np

errors = []

def chk(label, got, exp, cond=None):
    ok = (got == exp) if cond is None else cond
    tag = "OK" if ok else "XX"
    print(f"  [{tag}] {label:<40} got={got}  expect={exp}")
    if not ok:
        errors.append(label)

# ── 1. Metadata ──────────────────────────────────────────────────────────────
print("=== 3. METADATA SANITY ===")
with open("yolov5/runs/quant/extracted/metadata.json") as f: meta = json.load(f)
with open("yolov5/runs/quant/extracted/requant_params.json") as f: rq = json.load(f)
with open("golden/layer_shapes.json") as f: shapes = json.load(f)

total   = len(meta)
fp32    = [i for i in range(total) if meta[f"layer_{i:02d}"]["fp32_output"]]
rq_n    = len(rq)
all_in  = all(shapes[f"layer_{i:02d}"]["in"] is not None for i in range(total))
all_out = all(shapes[f"layer_{i:02d}"]["out"] is not None
              for i in range(total) if i not in fp32)

chk("Total layers",         total,   63)
chk("fp32-output count",    len(fp32), 4)
chk("fp32-output indices",  fp32, [42, 49, 56, 62])
chk("requant entries",      rq_n,    59)
chk("All in-shapes present",  all_in,  True)
chk("All out-shapes present", all_out, True)

# ── 2. N_config range ────────────────────────────────────────────────────────
print()
print("=== 4. N_CONFIG VALUES ===")
n_vals = {}
for i in range(total):
    p = f"weights/L{i:02d}_N_config.txt"
    with open(p) as f:
        n_vals[i] = int(f.read().strip())

n_arr    = list(n_vals.values())
all_valid = all(1 <= n <= 16 for n in n_arr)
n_dense   = sum(1 for n in n_arr if n == 16)
n_coo     = sum(1 for n in n_arr if 1 <= n <= 4)
n_bitmap  = sum(1 for n in n_arr if 5 <= n <= 15)

chk("All N in [1,16]",    all_valid, True)
chk("Dense (N=16) count", n_dense,   1)       # layer 62 only
chk("COO   (N<=4) count", n_coo,     62)
chk("Bitmap (5<=N<16)",   n_bitmap,  0)       # no bitmap in current model

# ── 3. Packed file sizes non-zero ────────────────────────────────────────────
print()
print("=== 5. PACKED FILE SIZES ===")
zero_packed = [i for i in range(total)
               if os.path.getsize(f"weights/L{i:02d}_packed.bin") == 0]
chk("Zero-size packed.bin files", len(zero_packed), 0, len(zero_packed)==0)

# ── 4. COO round-trip decode on 3 layers ─────────────────────────────────────
print()
print("=== 6. COO ROUND-TRIP (L01, L07, L25) ===")
M = 16

def decode_coo(path, n_path, shape):
    Cout, Cin, Kh, Kw = shape
    blen  = Cin * Kh * Kw
    padl  = (M - blen % M) % M
    pad   = blen + padl
    nblk  = pad // M
    with open(n_path) as f: N = int(f.read().strip())
    raw   = open(path, "rb").read()
    if N == M:
        return np.frombuffer(raw, dtype=np.int8).reshape(Cout, blen).reshape(Cout,Cin,Kh,Kw), N
    ipt   = (N+1)//2;  bpt = ipt + N
    total_blk = Cout * nblk;  grps = (total_blk+3)//4
    blocks = [];  pos = 0
    for _ in range(grps):
        for _ in range(4):
            if pos + bpt > len(raw):
                blocks.append(np.zeros(M, np.int8)); pos += bpt; continue
            nibs = []
            for b in raw[pos:pos+ipt]: nibs += [b&0xF, (b>>4)&0xF]
            idxs = nibs[:N]; pos += ipt
            vals = [struct.unpack("b", bytes([raw[pos+j]]))[0] for j in range(N)]; pos += N
            tile = np.zeros(M, np.int8)
            for i, v in zip(idxs, vals):
                if 0 <= i < M: tile[i] = v
            blocks.append(tile)
    wp = np.array(blocks[:total_blk], np.int8).reshape(Cout, pad)
    return wp[:, :blen].reshape(Cout, Cin, Kh, Kw), N

def pruned_ref(w_raw, shape, N):
    Cout, Cin, Kh, Kw = shape
    blen = Cin*Kh*Kw; padl = (M - blen%M)%M; pad = blen+padl; nblk = pad//M
    w = w_raw.reshape(Cout, blen).copy()
    if padl: w = np.pad(w, ((0,0),(0,padl)))
    for oc in range(Cout):
        for b in range(nblk):
            blk = w[oc, b*M:(b+1)*M].copy()
            zero_idx = np.argsort(np.abs(blk.astype(np.float32)), kind="stable")[:M-N]
            blk[zero_idx] = 0
            w[oc, b*M:(b+1)*M] = blk
    return w[:, :blen].reshape(Cout, Cin, Kh, Kw)

for li in [1, 7, 25]:
    key   = f"layer_{li:02d}"
    shape = meta[key]["weight_shape"]
    w_raw = np.frombuffer(
        open(f"yolov5/runs/quant/extracted/weights_bin/layer_{li:02d}.bin","rb").read(),
        dtype=np.int8)
    w_dec, N = decode_coo(f"weights/L{li:02d}_packed.bin",
                          f"weights/L{li:02d}_N_config.txt", shape)
    w_ref    = pruned_ref(w_raw, shape, N)
    diff     = int(np.sum(w_dec != w_ref))
    ok       = diff == 0
    chk(f"L{li:02d} ({meta[key]['name']}) N={N} round-trip",
        diff, 0, ok)

# ── 5. cnn_pkg.sv sanity ──────────────────────────────────────────────────────
print()
print("=== 7. CNN_PKG.SV SANITY ===")
pkg = open("rtl/pkg/cnn_pkg.sv").read()
for token in ["AUTO-GENERATED", "N_CFG", "M0_CH0", "SHIFT_CH0",
              "LAYER_CFG", "TOTAL_LAYERS", "AXI_BASE_WEIGHT"]:
    chk(f"contains '{token}'", token in pkg, True, token in pkg)

# ── 6. Summary ───────────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED: {len(errors)} check(s) failed:")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
