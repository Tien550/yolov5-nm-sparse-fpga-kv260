#!/usr/bin/env python3
"""Pack conv3x3 weights for B-c3 9-pass mode (Rev139, NS=12 all-SPEC).

conv3x3 = 9 tap-shifted conv1x1 passes on the unified 1x1 engine. Per (oc-block ob,
tap t, ic-chunk icc) the layer_sequencer DMAs ONE chunk; the engine's wide WT loader
(ST_WT_LOAD_W) consumes 64B-ALIGNED entries:

  chunk = 16 entries x 64B + requant 48x7B, padded to WT_BLK (1408B, 16-aligned)
  entry = [idx(1B, ASCENDING 0..15 — engine's c3_idxhw accumulator relies on this),
           48 weights (SPEC s, group g -> oc = ob*48 + s*4 + g), 15B pad]
  weight value = W[oc, icc*16 + idx, kh(t), kw(t)]
  requant tail at byte 1024: per oc 7B = [m0_lo, m0_hi, shift, bias b0..b3]

DDR chunk order: for ob: for t (0..8): for icc  — matches LS ic-loop (l_nicb = 9*NICC)
with wt DDR addr = wt_base + (ob*l_nicb + load_ic)*WT_BLK.

Layer set: 7 stride-verified conv3x3 (L01 s1 + L02/09/16/23/46/53 s2). L00 (Cin=3)
stays CPU (can't M-block a 3-channel input).

Usage: OUT_NAME=weights_unified_v4_ns12 python make_c3_weights.py
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARSE_DIR = os.environ.get('SPARSE_EXTRACTED',
    os.path.join(ROOT, 'yolov5/runs/quant/extracted_sparse'))
OUT_DIR = os.path.join(SPARSE_DIR, os.environ.get('OUT_NAME', 'weights_unified_v4_ns12'))

NS       = int(os.environ.get('NSPEC', '12'))
OCK      = NS * 4                 # output channels per kick (48 @NS=12)
ENTRY_SZ = 64                     # 64B-aligned entry (wide WT loader: 4 words, fixed routing)
RQ_BYTES = OCK * 7

# Sparse-c3 (Rev150): C3_IDXMAP = c3_idx_map.json từ finetune {module_name: [[idx/mblk], ...]}.
# Lớp có trong map → pack chỉ N kept-idx/M-block (N=len); engine đọc idx từ entry, addressing
# ACT = wt_rd_idx[slot]*HW. Lớp không có trong map → dense N=16 (giữ nguyên). N uniform/lớp.
C3_IDXMAP = os.environ.get('C3_IDXMAP')
_IDXMAP   = json.load(open(C3_IDXMAP)) if C3_IDXMAP else {}


def blk_bytes(n_cfg):
    """WT_BLK cho N_CFG entries: N*64 + requant(OCK*7), pad 16. Requant tại offset N*64
    (engine: wt_nz_bytes_c3 = reg_n_config<<6)."""
    raw = n_cfg * ENTRY_SZ + RQ_BYTES
    return raw, ((raw + 15) // 16) * 16


def make_c3_chunk(W, m0, sh, bias, ob, t, icc, kept):
    """One (ob, tap, icc) chunk. kept = danh sách idx (0..15) giữ lại cho M-block icc
    (ascending). Dense: kept = range(16). W: (Cout, Cin, 3, 3) int8."""
    kh, kw = t // 3, t % 3
    Cout = W.shape[0]
    n_cfg = len(kept)
    c3_raw, c3_blk = blk_bytes(n_cfg)
    blob = bytearray()
    for idx in kept:                           # idx = kept channel offset trong M-block
        ic = icc * 16 + idx
        entry = bytearray([idx])
        for spec in range(NS):
            for g in range(4):
                oc = ob * OCK + spec * 4 + g
                entry.append(int(W[oc, ic, kh, kw]) & 0xFF if oc < Cout else 0)
        entry += bytes(ENTRY_SZ - len(entry))
        blob += entry
    for local in range(OCK):
        oc = ob * OCK + local
        if oc < Cout:
            m0v = int(m0[oc]) & 0xFFFF; shv = int(sh[oc]) & 0x1F; bv = int(bias[oc])
        else:
            m0v, shv, bv = 0x4000, 14, 0
        blob += bytes([m0v & 0xFF, (m0v >> 8) & 0xFF, shv,
                       bv & 0xFF, (bv >> 8) & 0xFF, (bv >> 16) & 0xFF, (bv >> 24) & 0xFF])
    assert len(blob) == c3_raw
    return bytes(blob) + bytes(c3_blk - c3_raw)


def kept_for_layer(name, NICC):
    """Trả (kept_per_icc list, N_CFG). Sparse nếu name trong C3_IDXMAP, else dense 16.
    idx_map từ finetune đã sort ascending; assert đủ NICC M-block & N uniform."""
    if name not in _IDXMAP:
        return [list(range(16)) for _ in range(NICC)], 16
    km = _IDXMAP[name]
    assert len(km) == NICC, f'{name}: idx_map {len(km)} M-block != NICC {NICC}'
    ns = {len(k) for k in km}
    assert len(ns) == 1, f'{name}: N không uniform per M-block: {ns}'
    kept = [sorted(int(i) for i in k) for k in km]
    return kept, len(kept[0])


def main():
    meta = json.load(open(os.path.join(SPARSE_DIR, 'metadata.json')))
    req  = json.load(open(os.path.join(SPARSE_DIR, 'requant_params.json')))
    os.makedirs(OUT_DIR, exist_ok=True)

    c3_params = {}
    for lid in range(63):
        k = f'layer_{lid:02d}'
        m = meta.get(k)
        if m is None:
            continue
        Cout, Cin, kH, kW = m['weight_shape']
        if kH != 3 or kW != 3 or m.get('is_depthwise', False):
            continue
        if k not in req:
            continue

        W  = np.fromfile(os.path.join(SPARSE_DIR, 'weights_bin', f'{k}.bin'),
                         dtype=np.int8).reshape(Cout, Cin, kH, kW)
        Cin_real = Cin
        if Cin % 16 != 0:
            # L00 lên HW (Cin=3): pad kênh → 16 với weights 0. Plane act chỉ chứa Cin_real
            # kênh; engine đọc idx ≥ Cin_real trúng rác act_uram × weight 0 = 0 → EXACT
            # (cùng chứng minh pad-OC/chain). 'Cin' trong json giữ SỐ THẬT cho host assert.
            Cin_pad = ((Cin + 15) // 16) * 16
            Wp = np.zeros((Cout, Cin_pad, kH, kW), dtype=np.int8)
            Wp[:, :Cin] = W
            W = Wp; Cin = Cin_pad
        b  = np.fromfile(os.path.join(SPARSE_DIR, 'bias_bin', f'{k}.bin'), dtype=np.int32)
        m0 = np.array(req[k]['M0'],    dtype=np.uint16)
        sh = np.array(req[k]['shift'], dtype=np.uint8)

        NICC   = Cin // 16                      # ic-chunks per tap (CIN=16/kick)
        N_ICB  = 9 * NICC                       # LS ic-loop length
        N_OCB  = (Cout + OCK - 1) // OCK
        stride = m['stride'][0]

        # Sparse-c3: kept idx per M-block + N_CFG (dense 16 nếu lớp không trong idx_map).
        kept_list, N_CFG = kept_for_layer(m['name'], NICC)
        c3_raw, C3_BLK = blk_bytes(N_CFG)

        out_path = os.path.join(OUT_DIR, f'c3_{k}.bin')
        with open(out_path, 'wb') as f:
            for ob in range(N_OCB):
                for t in range(9):
                    for icc in range(NICC):
                        f.write(make_c3_chunk(W, m0, sh, b, ob, t, icc, kept_list[icc]))

        wt_total = N_OCB * N_ICB * C3_BLK
        sp = f' SPARSE N={N_CFG}' if m['name'] in _IDXMAP else ' dense'
        c3_params[k] = {
            'type':      'c3',
            'Cin':       Cin_real,
            'Cout':      Cout,
            'NICC':      NICC,
            'N_ICB':     N_ICB,
            'N_OCB':     N_OCB,
            'N_CFG':     N_CFG,
            'NSPEC':     NS,
            'OC_PER_KICK': OCK,
            'stride':    stride,
            'WT_BLK':    C3_BLK,
            'wt_total':  wt_total,
        }
        print(f'{k}: c3 Cout={Cout} Cin={Cin} s={stride} N_OCB={N_OCB} N_ICB={N_ICB} '
              f'N_CFG={N_CFG}{sp} WT_BLK={C3_BLK} total={wt_total/1024:.0f}KB  -> {out_path}')

    json.dump(c3_params, open(os.path.join(OUT_DIR, 'c3_layer_params.json'), 'w'), indent=2)
    total = sum(p['wt_total'] for p in c3_params.values())
    max_ob_uram = max(p['N_ICB'] * p['WT_BLK'] for p in c3_params.values())
    print(f'\nDone. {len(c3_params)} conv3x3 layers -> {OUT_DIR}')
    print(f'Total c3 weight size: {total/1024/1024:.2f}MB. '
          f'Max per-ob URAM = {max_ob_uram/1024:.0f}KB (wt_uram 256KB)')


if __name__ == '__main__':
    main()
