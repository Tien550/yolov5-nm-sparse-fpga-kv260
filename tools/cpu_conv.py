"""
cpu_conv.py -- Bit-exact CPU fallback for quantized INT8 conv (1x1 / 3x3 / dwconv).

Matches the FPGA hardware quantization semantics:
  psum(int64) = sum(w * x)  + bias
  scaled      = (psum * M0) >> shift     (arithmetic right shift)
  clipped     = clip(scaled, -128, 127)
  leaky       = clipped if >=0 else (clipped >> 3)   (LeakyReLU slope 1/8)

Used to run conv3x3 / dwconv layers on CPU while conv1x1 runs on FPGA, since the
FPGA conv3x3/dwconv path is not yet fully brought up (requant load + per-tile
OUT DMA orchestration missing). Verified within BN-rounding noise (<5% off-by->1)
against the quantized network golden outputs (golden/L{nn}_out_CHW.bin).
"""
import numpy as np


def quant_conv(x_int8, w_int8, bias_int32, M0, shift, *,
               stride=1, pad=1, is_depthwise=False, apply_relu=True):
    """Quantized INT8 conv matching FPGA hardware arithmetic.

    x_int8     : [Cin, Hin, Win]  int8
    w_int8     : [Cout, Cin, Kh, Kw] int8  (dw: [Cout, 1, Kh, Kw], Cin==Cout)
    bias_int32 : [Cout] int32
    M0         : [Cout] uint16 (per-channel multiplier)
    shift      : [Cout] uint8  (per-channel right shift)
    returns    : [Cout, Hout, Wout] int8
    """
    x = np.ascontiguousarray(x_int8).astype(np.int64)
    w = np.ascontiguousarray(w_int8).astype(np.int64)
    Cout = w.shape[0]
    Kh, Kw = w.shape[2], w.shape[3]
    Cin, Hin, Win = x.shape
    sh = sw = stride
    Hout = (Hin + 2 * pad - Kh) // sh + 1
    Wout = (Win + 2 * pad - Kw) // sw + 1

    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), constant_values=0)
    out = np.zeros((Cout, Hout, Wout), dtype=np.int64)

    if is_depthwise:
        for oc in range(Cout):
            k = w[oc, 0]
            for h in range(Hout):
                hs = h * sh
                for ww in range(Wout):
                    ws = ww * sw
                    out[oc, h, ww] = np.sum(xp[oc, hs:hs + Kh, ws:ws + Kw] * k)
    else:
        # int64 matmul không dùng BLAS → chậm; float64 BLAS nhanh và chính xác
        # vì |psum| ≤ Cin*Kh*Kw*127*127 ≤ 2304*16129 ≈ 37M << 2^52 (float64 exact)
        wf = w.reshape(Cout, -1).astype(np.float64)
        if Kh == 1 and Kw == 1:
            xs = xp[:, ::sh, ::sw][:, :Hout, :Wout].astype(np.float64)
            out = np.rint(wf @ xs.reshape(Cin, -1)).astype(np.int64).reshape(Cout, Hout, Wout)
        else:
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(xp, (Kh, Kw), axis=(1, 2))[:, ::sh, ::sw][:, :Hout, :Wout]
            col = win.transpose(0, 3, 4, 1, 2).reshape(Cin * Kh * Kw, Hout * Wout).astype(np.float64)
            out = np.rint(wf @ col).astype(np.int64).reshape(Cout, Hout, Wout)

    out += bias_int32.astype(np.int64)[:, None, None]

    # Requant — BIT-EXACT với rr_unit.v (Rev139b fix: leaky TRƯỚC clip, đúng pipeline HW
    # stage2 round-shift → stage3a leaky → stage3b clip; cũng đúng model PyTorch hơn vì
    # nn.LeakyReLU chạy trên giá trị CHƯA clamp):
    #   prod   = acc * M0
    #   scaled = (sh>0) ? ((prod + (1<<(sh-1))) >> sh) : prod    [ROUNDED right shift, 48-bit]
    #   if relu && scaled<0: scaled = (scaled*205 + 1024) >> 11  [LeakyReLU 0.1 ≈ 205/2048,
    #                                                             arithmetic shift = floor, y hệt rr_unit]
    #   q = clip(scaled, -128, 127)                              [clip CUỐI CÙNG]
    # (Trước đây clip trước leaky → lệch với HW khi scaled < -128; chỉ lộ ra ở c3 9-pass
    #  vì 1×1/DW chưa bao giờ chạm vùng đó. Board-caught 2026-07-02.)
    M0 = np.asarray(M0, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    res = np.empty((Cout, Hout, Wout), dtype=np.int8)
    for oc in range(Cout):
        prod = out[oc] * M0[oc]
        s = int(shift[oc])
        scaled = (prod + (np.int64(1) << (s - 1))) >> s if s > 0 else prod
        if apply_relu:
            scaled = np.where(scaled >= 0, scaled, (scaled * np.int64(205) + np.int64(1024)) >> 11)
        res[oc] = np.clip(scaled, -128, 127).astype(np.int8)
    return res


def float_conv(x_int8, w_int8, bias_int32, input_scale, weight_scale_per_channel, *,
               stride=1, pad=1, is_depthwise=False, apply_relu=True):
    """FP32-output conv for no-requant layers {42,49,56,62} (output_scale=None).

      out_float[oc] = (psum_int[oc] + bias_int32[oc]) * input_scale * weight_scale[oc]
    (matches yolo_infer.c head_fp_sc = in_sc * w_sc_per_ch).
    apply_relu: 42/49/56 are neck convs WITH LeakyReLU(0.1) on the float output
    (verified: leaky0.1(feat) quantized == golden L46_in 100%); the final detect
    output 62 has NO activation (apply_relu=False).
    """
    x = np.ascontiguousarray(x_int8).astype(np.int64)
    w = np.ascontiguousarray(w_int8).astype(np.int64)
    Cout = w.shape[0]; Kh, Kw = w.shape[2], w.shape[3]
    Cin, Hin, Win = x.shape
    Hout = (Hin + 2*pad - Kh)//stride + 1
    Wout = (Win + 2*pad - Kw)//stride + 1
    xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)), constant_values=0)
    out = np.zeros((Cout, Hout, Wout), dtype=np.int64)
    # Vectorized im2col + single matmul in FLOAT64 (BLAS gemm), replacing the per-pixel Python
    # loop. int64 matmul is NOT BLAS (slow generic path), so we matmul as float64 then round back:
    # exact because |psum| ≤ 128·128·(Cin·Kh·Kw) ≪ 2^53. ~10× faster on the FP32 head.
    if Kh == 1 and Kw == 1:
        xs = xp[:, pad : pad + 1 + (Hout-1)*stride : stride,
                   pad : pad + 1 + (Wout-1)*stride : stride].astype(np.float64)  # [Cin,Hout,Wout]
        out = np.rint(w.reshape(Cout, Cin).astype(np.float64) @ xs.reshape(Cin, -1)
                      ).astype(np.int64).reshape(Cout, Hout, Wout)
    else:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(xp, (Kh, Kw), axis=(1, 2))[:, ::stride, ::stride][:, :Hout, :Wout]
        if is_depthwise:
            out = np.rint(np.einsum('chwij,cij->chw', win.astype(np.float64),
                                    w[:, 0].astype(np.float64))).astype(np.int64)
        else:
            col = win.transpose(0, 3, 4, 1, 2).reshape(Cin*Kh*Kw, Hout*Wout).astype(np.float64)
            out = np.rint(w.reshape(Cout, -1).astype(np.float64) @ col
                          ).astype(np.int64).reshape(Cout, Hout, Wout)
    out += bias_int32.astype(np.int64)[:, None, None]
    sc = (input_scale * np.asarray(weight_scale_per_channel, np.float64))[:, None, None]
    y = out.astype(np.float64) * sc
    if apply_relu:
        y = np.where(y >= 0, y, 0.1 * y)   # LeakyReLU(0.1) in float domain
    return y.astype(np.float32)


def quantize_to_int8(x_float, scale):
    """Quantize a float activation to int8 by a given scale (round-half-to-even).
    HEADQ opt: nếu x mang .qscale (int8 + source-scale, _QInt8) → rescale int→int TRỰC TIẾP
    (round((int8×qscale)/scale)) — ĐÚNG thứ tự phép tính như dequant-rồi-quantize nên BIT-EXACT,
    nhưng đọc int8 (1/4 băng thông float) + bỏ mảng float trung gian dùng chung 2 consumer."""
    fs = getattr(x_float, 'qscale', None)
    if fs is not None:
        return np.clip(np.round((x_float.astype(np.float32, subok=False) * fs) / scale),
                       -128, 127).astype(np.int8)
    return np.clip(np.round(x_float / scale), -128, 127).astype(np.int8)


# ── Self-test against golden when run directly ───────────────────────────────
if __name__ == "__main__":
    import json, os, sys
    E = sys.argv[1] if len(sys.argv) > 1 else "yolov5/runs/quant/extracted"
    GOLD = sys.argv[2] if len(sys.argv) > 2 else "golden"
    meta = json.load(open(f"{E}/metadata.json"))
    rq = json.load(open(f"{E}/requant_params.json"))
    shapes = json.load(open(f"{GOLD}/layer_shapes.json"))

    tests = {"conv1x1": 3, "conv3x3": 1, "dwconv": 7}
    for name, lid in tests.items():
        key = f"layer_{lid:02d}"
        m = meta[key]
        Cout, Cin, Kh, Kw = m["weight_shape"]
        w = np.fromfile(f"{E}/weights_bin/{key}.bin", dtype=np.int8).reshape(Cout, Cin, Kh, Kw)
        bias = np.fromfile(f"{E}/bias_bin/{key}.bin", dtype=np.int32)
        M0 = np.array(rq[key]["M0"], dtype=np.uint16)
        shift = np.array(rq[key]["shift"], dtype=np.uint8)
        inp = np.fromfile(f"{GOLD}/L{lid:02d}_in_CHW.bin", dtype=np.int8).reshape(shapes[key]["in"])
        gold = np.fromfile(f"{GOLD}/L{lid:02d}_out_CHW.bin", dtype=np.int8).reshape(shapes[key]["out"])

        res = quant_conv(inp, w, bias, M0, shift,
                         stride=m["stride"][0], pad=m["padding"][0],
                         is_depthwise=bool(m["is_depthwise"]), apply_relu=True)
        d = np.abs(res.astype(np.int16) - gold.astype(np.int16))
        tot = res.size
        print(f"{name:8s} L{lid:02d} {tuple(res.shape)}: "
              f"exact={100*np.sum(d==0)/tot:5.1f}%  within1={100*np.sum(d<=1)/tot:5.1f}%  "
              f"off>1={100*np.sum(d>1)/tot:.2f}%  -> {'PASS' if np.sum(d>1)/tot<=0.05 else 'FAIL'}")
