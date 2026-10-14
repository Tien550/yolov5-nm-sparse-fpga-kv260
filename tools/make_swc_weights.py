#!/usr/bin/env python3
"""Pack depthwise (DW) conv weights for unified_acc_top swc FSM.

DW weight blob format per OCB (4 output channels ob*4..ob*4+3):
  9 taps × 17B: [idx=0, W[g0,kh,kw], W[g1,kh,kw], W[g2,kh,kw], W[g3,kh,kw], zeros×12]
  4 × 7B requant: [m0_lo, m0_hi, rq_shift, b0..b3]
    rq_shift = req['shift'][ch] written directly (already total shift for rr_unit,
    same convention as make_unified_weights.py for conv1x1).

Output: weights_unified_v4/dw_layer_XX.bin + dw_layer_params.json
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARSE_DIR = os.environ.get('SPARSE_EXTRACTED',
    os.path.join(ROOT, 'yolov5/runs/quant/extracted_sparse'))
OUT_DIR = os.path.join(SPARSE_DIR, os.environ.get('OUT_NAME', 'weights_unified_v4'))

# NSPEC (Rev139): RTL entry = 1 + NS*4 bytes (NS = N_SPEC_ACT in cnn_defines.vh).
# DW dùng SPEC0 only → weights SPEC1..NS-1 = 0. NSPEC=0/unset → legacy NS=4 (entry 17B).
NSPEC = int(os.environ.get('NSPEC', '0'))
_NS   = NSPEC if NSPEC > 0 else 4
DW_ESZ    = 1 + _NS * 4                    # entry bytes: idx + NS*4 weights
DW_WT_RAW = 9 * DW_ESZ + 4 * 7             # 9 taps × ESZ + 4×7B requant
DW_WT_BLK = ((DW_WT_RAW + 15) // 16) * 16  # padded to 16B boundary


def make_dw_blob(W, m0, sh, bias, ob):
    """Pack one DW OCB (ob) blob = DW_WT_BLK bytes.

    W:    (N_CH, 3, 3) int8
    m0:   (N_CH,) uint16
    sh:   (N_CH,) uint8  — total rq_shift (NOT extra shift), written directly
    bias: (N_CH,) int32
    ob:   OCB index, processes channels ob*4..ob*4+3
    """
    blob = bytearray()
    for kh in range(3):
        for kw in range(3):
            entry = bytearray([0])  # idx=0 (dense; tap=kh*3+kw used implicitly by FSM)
            for g in range(4):
                entry.append(int(W[ob * 4 + g, kh, kw]) & 0xFF)
            entry += bytes(DW_ESZ - 5)  # SPEC 1..NS-1 zeros
            assert len(entry) == DW_ESZ
            blob += entry
    for g in range(4):
        ch = ob * 4 + g
        m0v = int(m0[ch]) & 0xFFFF
        shv = int(sh[ch]) & 0x1F
        bv  = int(bias[ch])
        blob += bytes([m0v & 0xFF, (m0v >> 8) & 0xFF, shv,
                       bv & 0xFF, (bv >> 8) & 0xFF, (bv >> 16) & 0xFF, (bv >> 24) & 0xFF])
    assert len(blob) == DW_WT_RAW
    return bytes(blob) + bytes(DW_WT_BLK - DW_WT_RAW)


def main():
    meta = json.load(open(os.path.join(SPARSE_DIR, 'metadata.json')))
    req  = json.load(open(os.path.join(SPARSE_DIR, 'requant_params.json')))
    os.makedirs(OUT_DIR, exist_ok=True)

    dw_params = {}

    for lid in range(63):
        k = f'layer_{lid:02d}'
        m = meta.get(k)
        if m is None:
            continue
        ws   = m['weight_shape']
        Cout, Cin, kH, kW = ws
        isdw = bool(m.get('is_depthwise', False))

        if not isdw or kH != 3 or kW != 3:
            continue
        if Cout % 4 != 0:
            continue
        if k not in req:
            continue

        # DW weight stored as (Cout, 1, kH, kW) — reshape to (Cout, kH, kW)
        W = np.fromfile(os.path.join(SPARSE_DIR, 'weights_bin', f'{k}.bin'),
                        dtype=np.int8).reshape(Cout, kH, kW)
        b  = np.fromfile(os.path.join(SPARSE_DIR, 'bias_bin', f'{k}.bin'),
                         dtype=np.int32)
        m0 = np.array(req[k]['M0'],    dtype=np.uint16)
        sh = np.array(req[k]['shift'], dtype=np.uint8)
        N_OCB = Cout // 4

        out_path = os.path.join(OUT_DIR, f'dw_{k}.bin')
        with open(out_path, 'wb') as f:
            for ob in range(N_OCB):
                f.write(make_dw_blob(W, m0, sh, b, ob))

        wt_total = N_OCB * DW_WT_BLK
        stride = m.get('stride', [1])[0]
        dw_params[k] = {
            'type':     'dw',
            'N_CH':     Cout,
            'N_OCB':    N_OCB,
            'N_ICB':    1,
            'N_CFG':    1,
            'NSPEC':    _NS,
            'stride':   stride,
            'WT_BLK':   DW_WT_BLK,
            'wt_total': wt_total,
        }
        print(f'{k}: DW Cout={Cout} stride={stride} N_OCB={N_OCB} '
              f'WT_BLK={DW_WT_BLK} total={wt_total}B  -> {out_path}')

    json.dump(dw_params, open(os.path.join(OUT_DIR, 'dw_layer_params.json'), 'w'), indent=2)
    total = sum(p['wt_total'] for p in dw_params.values())
    print(f'\nDone. {len(dw_params)} DW layers -> {OUT_DIR}')
    print(f'Total DW weight size: {total}B ({total/1024:.1f}KB)')
    print(f'DW_WT_BLK={DW_WT_BLK}B (DW_WT_RAW={DW_WT_RAW}B)')


if __name__ == '__main__':
    main()
