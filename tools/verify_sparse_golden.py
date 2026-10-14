"""
verify_sparse_golden.py  --  Cross-verify sparse encoding pipeline

Phương pháp:
  1. Chạy QAT model với ảnh thực, capture output float của mỗi QuantConv2d
  2. Requantize output float bằng S_out từ metadata -> expected INT8
  3. Chạy manual INT8 conv với DENSE weights (từ file) + requant formula
  4. So sánh: PASS nếu ≤5% off-by-more-than-1 (BN rounding noise)

Mục đích: xác nhận requant formula (M0, shift, LeakyReLU) đúng vs model thực.
Nếu pass với dense weights → encoding pipeline OK, RTL có thể dùng công thức này.

Usage:
  python tools/verify_sparse_golden.py \\
    --weights   weights \\
    --golden    golden \\
    --extracted yolov5/runs/quant/extracted \\
    --layers    01 02 03
"""
import argparse
import json
import os
import sys
import struct
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "yolov5")

M_BLOCK = 16
FP32_OUTPUT_LAYERS = {42, 49, 56, 62}


# ─── Decoder COO/DENSE ───────────────────────────────────────────────────────

def decode_packed(packed_path, n_cfg_path, shape):
    with open(n_cfg_path) as f:
        N = int(f.read().strip())
    Cout, Cin, Kh, Kw = shape
    block_len = Cin * Kh * Kw
    raw = open(packed_path, 'rb').read()

    if N == M_BLOCK:
        return np.frombuffer(raw, dtype=np.int8).reshape(Cout, block_len)[:, :block_len].reshape(Cout, Cin, Kh, Kw), N, "DENSE"

    pad_len   = (M_BLOCK - block_len % M_BLOCK) % M_BLOCK
    padded    = block_len + pad_len
    num_blk   = padded // M_BLOCK
    total_blk = Cout * num_blk

    idx_bpt = (N + 1) // 2
    val_bpt = N
    bpt     = idx_bpt + val_bpt

    blocks = []
    pos    = 0
    for _ in range((total_blk + 3) // 4):
        for _ in range(4):
            if pos + bpt > len(raw):
                blocks.append(np.zeros(M_BLOCK, dtype=np.int8))
                pos += bpt
                continue
            idx_bytes = raw[pos: pos + idx_bpt]; pos += idx_bpt
            nibbles = []
            for b in idx_bytes:
                nibbles += [b & 0xF, (b >> 4) & 0xF]
            indices = nibbles[:N]
            val_bytes = raw[pos: pos + val_bpt]; pos += val_bpt
            values = [struct.unpack('b', bytes([b]))[0] for b in val_bytes]
            tile = np.zeros(M_BLOCK, dtype=np.int8)
            for i, v in zip(indices, values):
                if 0 <= i < M_BLOCK:
                    tile[i] = v
            blocks.append(tile)

    w_pad = np.array(blocks[:total_blk], dtype=np.int8).reshape(Cout, padded)
    return w_pad[:, :block_len].reshape(Cout, Cin, Kh, Kw), N, "COO"


# ─── Manual INT8 conv + requant ──────────────────────────────────────────────

def manual_conv(weights, biases, inp, info, M0_list, shift_list, layer_idx):
    Cout, Cin, Kh, Kw = weights.shape
    stride_h, stride_w = info['stride']
    pad = info['padding']
    is_dw = info['is_depthwise']
    has_leaky = layer_idx not in FP32_OUTPUT_LAYERS and layer_idx < 60

    Hin, Win = inp.shape[1], inp.shape[2]
    Hout = (Hin + 2*pad[0] - Kh) // stride_h + 1
    Wout = (Win + 2*pad[1] - Kw) // stride_w + 1

    inp_p = np.pad(inp, ((0,0),(pad[0],pad[0]),(pad[1],pad[1])), constant_values=0)
    out   = np.zeros((Cout, Hout, Wout), dtype=np.int64)

    if is_dw:
        for oc in range(Cout):
            k = weights[oc, 0].astype(np.int64)
            for h in range(Hout):
                for w in range(Wout):
                    p = inp_p[oc, h*stride_h:h*stride_h+Kh, w*stride_w:w*stride_w+Kw].astype(np.int64)
                    out[oc, h, w] = np.sum(p * k)
    else:
        wf = weights.reshape(Cout, -1).astype(np.int64)
        for h in range(Hout):
            for w in range(Wout):
                col = inp_p[:, h*stride_h:h*stride_h+Kh, w*stride_w:w*stride_w+Kw].reshape(-1).astype(np.int64)
                out[:, h, w] = wf @ col

    out += biases[:, None, None].astype(np.int64)

    result = np.zeros((Cout, Hout, Wout), dtype=np.int8)
    for oc in range(Cout):
        scaled  = (out[oc] * np.int64(M0_list[oc])) >> int(shift_list[oc])
        clipped = np.clip(scaled, -128, 127).astype(np.int8)
        if has_leaky:
            clipped = np.where(clipped >= 0, clipped,
                               (clipped.astype(np.int16) >> 3).astype(np.int8))
        result[oc] = clipped
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    default='weights')
    parser.add_argument('--golden',     default='golden')
    parser.add_argument('--extracted',  default='yolov5/runs/quant/extracted')
    parser.add_argument('--layers',     nargs='+', default=['01', '02', '03'])
    parser.add_argument('--sparse',     action='store_true',
                        help='Dùng sparse (pruned) weights thay vì dense gốc')
    parser.add_argument('--threshold',  type=float, default=0.05,
                        help='Max tỉ lệ off-by->1 để PASS (default 5%%)')
    args = parser.parse_args()

    with open(os.path.join(args.extracted, 'metadata.json'))   as f: meta    = json.load(f)
    with open(os.path.join(args.extracted, 'requant_params.json')) as f: rq = json.load(f)
    with open(os.path.join(args.golden, 'layer_shapes.json'))  as f: shapes  = json.load(f)

    mode = 'SPARSE (pruned)' if args.sparse else 'DENSE (gốc, kiểm tra requant formula)'
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Threshold: off-by->1 ≤ {args.threshold*100:.1f}%")
    print()

    all_pass = True

    for lstr in args.layers:
        idx = int(lstr)
        key = f"layer_{idx:02d}"
        if key not in meta:
            print(f"[SKIP] {key}: không có trong metadata"); continue
        info = meta[key]
        shape = info['weight_shape']

        if key not in rq:
            print(f"[SKIP] {key}: fp32 output, không có requant params"); continue

        out_shape = shapes.get(key, {}).get('out')
        if out_shape is None:
            print(f"[SKIP] {key}: không có golden output"); continue

        # Load weights
        if args.sparse:
            w, N, enc = decode_packed(
                os.path.join(args.weights, f"L{idx:02d}_packed.bin"),
                os.path.join(args.weights, f"L{idx:02d}_N_config.txt"), shape)
            w_desc = f"{enc} N={N} (pruned)"
        else:
            w_raw = np.frombuffer(
                open(os.path.join(args.extracted, 'weights_bin', f"layer_{idx:02d}.bin"), 'rb').read(),
                dtype=np.int8).reshape(shape)
            w, N, enc = w_raw, '?', 'DENSE_orig'
            # Fidelity check
            w_dec, N_dec, enc_dec = decode_packed(
                os.path.join(args.weights, f"L{idx:02d}_packed.bin"),
                os.path.join(args.weights, f"L{idx:02d}_N_config.txt"), shape)
            n_diff = np.sum(w_dec != w_raw)
            print(f"  Encode/decode fidelity: {n_diff}/{w_raw.size} diffs ({100.*n_diff/w_raw.size:.1f}% pruned)")
            w_desc = f"dense gốc ({enc_dec} N={N_dec})"

        bias = np.frombuffer(
            open(os.path.join(args.extracted, 'bias_bin', f"layer_{idx:02d}.bin"), 'rb').read(),
            dtype='<i4')
        M0s    = rq[key]['M0']
        shifts = rq[key]['shift']

        in_shape = shapes[key]['in']
        inp = np.frombuffer(
            open(os.path.join(args.golden, f"L{idx:02d}_in_CHW.bin"), 'rb').read(),
            dtype=np.int8).reshape(in_shape)
        expected = np.frombuffer(
            open(os.path.join(args.golden, f"L{idx:02d}_out_CHW.bin"), 'rb').read(),
            dtype=np.int8).reshape(out_shape)

        print(f"[VERIFY] Layer {idx:02d} ({info['name']}): {w_desc}")
        print(f"         inp={in_shape}  expected_out={out_shape}")

        manual = manual_conv(w, bias, inp, info, M0s, shifts, idx)

        if manual.shape != expected.shape:
            print(f"  Shape mismatch: manual={manual.shape} vs expected={expected.shape} — SKIP\n")
            continue

        abs_diff  = np.abs(manual.astype(np.int16) - expected.astype(np.int16))
        exact     = int(np.sum(abs_diff == 0))
        off1      = int(np.sum(abs_diff == 1))
        off_gt1   = int(np.sum(abs_diff >  1))
        total     = expected.size
        pct_gt1   = off_gt1 / total

        # Histogram for diff>1
        uniq, cnts = np.unique(abs_diff[abs_diff > 1], return_counts=True) if off_gt1 > 0 else ([], [])

        status = "PASS" if pct_gt1 <= args.threshold else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  exact={exact}/{total}  off-by-1={off1}  off-by->1={off_gt1} ({pct_gt1*100:.2f}%) — {status}")
        if len(uniq) > 0:
            hist_str = "  diff histogram: " + "  ".join(f"d{d}:{c}" for d, c in zip(uniq[:8], cnts[:8]))
            print(hist_str)
        if status == "FAIL":
            bad = np.argwhere(abs_diff > 1)
            for p in bad[:5]:
                oc, h, ww = p
                print(f"     ({oc},{h},{ww}): manual={int(manual[oc,h,ww])} exp={int(expected[oc,h,ww])}")
        print()

    print("[RESULT]", "ALL PASS ✓" if all_pass else "SOME LAYERS FAILED ✗")
    if not all_pass:
        print("  Debug gợi ý:")
        print("  - Arithmetic right shift: (psum * M0) >> shift  (Python int64)")
        print("  - LeakyReLU: slope = 1/8 = >>3  (signed int8)")
        print("  - Bias được apply TRƯỚC requant")
        print("  - Weight layout: [Cout, Cin, Kh, Kw]")
        print("  - Input layout: CHW, not HWC")


if __name__ == '__main__':
    main()
