#!/usr/bin/env python3
"""Convert extracted_sparse dense INT8 weights to unified_acc_top COO blobs.

Input:  yolov5/runs/quant/extracted_sparse/
          metadata.json, requant_params.json, weights_bin/, bias_bin/
Output: yolov5/runs/quant/extracted_sparse/weights_unified_v4/
          layer_XX.bin        - N_OCB x N_IC_BLKS blobs, each WT_BLK=80 bytes
          layer_params.json   - per layer: WT_BLK, N_OCB, N_IC_BLKS, CIN, COUT, ...
          nz_map.json         - NZ indices per layer per M-block (for verification)

Blob format per (oc-block, M-block) -- poc=0 mode, SPEC 0 only:
  For each NZ slot k in 0..N_CFG-1:
    [idx(1B), s0g0,s0g1,s0g2,s0g3, s1..s3 all zeros (12B)] = 17B total
  Then poc_size=4 entries of 7B requant: [m0_lo, m0_hi, sh, bias_b0..b3]
    local 0..3 = oc_block*4+0..3 (real values for the 4 active output channels)
  Padded to WT_BLK=80B (fixed for all layers: N_CFG=3, poc_size=4).

DDR layout: blob[ob][mt] at offset (ob*N_IC_BLKS + mt)*WT_BLK from layer base.
Matches layer_seq: wt_load_addr = wt_base + (oc*N_ICB + load_ic)*WT_BLK_BYTES

NZ selection: top N_CFG positions per M-block by summed |weight| across all output channels.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARSE_DIR = os.environ.get(
    'SPARSE_EXTRACTED',
    os.path.join(ROOT, 'yolov5/runs/quant/extracted_sparse'))
OUT_DIR  = os.path.join(SPARSE_DIR, os.environ.get('OUT_NAME', 'weights_unified_v4'))
N_CFG    = 3   # NZ entries per M-block (mặc định uniform — paper N=3/16)
# NCFG_AUTO=1 (DominoSearch non-uniform): N per-layer suy từ CHÍNH weights int8 — đếm cột
# nonzero (chia sẻ mọi oc) mỗi M-block, lấy max qua các block. Model lw đã bake mask nên
# đếm ra đúng scheme; blob/WT_BLK/layer_params tự theo (host đọc N_CFG per-layer sẵn).
NCFG_AUTO = bool(int(os.environ.get('NCFG_AUTO', '0')))
# FORCE_N (2026-07-10, thí nghiệm dense-encoded A/B): ép N_CFG cố định cho MỌI layer,
# bỏ qua NCFG_AUTO. FORCE_N=16 = dense encode (pack đủ 16 entry/block kể cả weight 0)
# → cùng weights, cùng kết quả bit-exact, chỉ TIMING đổi — đo cổ tức sparsity trên board.
FORCE_N = int(os.environ.get('FORCE_N', '0'))
if FORCE_N > 0:
    N_CFG = FORCE_N
    NCFG_AUTO = False
POC_SIZE = 4   # legacy poc=0 mode: 4 output channels per kick (SPEC 0 only)
GROUPS   = 4   # G=4 groups per SPEC

# ── B-w mode: NSPEC env selects the number of SPEC filled (output-parallel) ──────
#   NSPEC unset/0 → LEGACY (SPEC0-only, 4 OC/kick, WT_BLK=80) — for Rev135/Rev136 NS=4 baseline.
#   NSPEC=N (>0)  → fill ALL N SPEC: SPEC s, group g → oc = ob*(N*4) + s*4 + g. N*4 OC/kick.
#                   Entry = 1 idx + N*4 weights; requant tail = N*4 entries. WT_BLK padded.
#   MUST match RTL `N_SPEC_ACT` in cnn_defines.vh and inference REG_COUT=N*4, N_OCB=ceil(Cout/(N*4)).
NSPEC = int(os.environ.get('NSPEC', '0'))

# Legacy fixed blob size (N_CFG=3, POC_SIZE=4): N_CFG*17 + POC_SIZE*7 = 79 -> 80B
_WT_RAW = N_CFG * 17 + POC_SIZE * 7   # 79
WT_BLK  = ((_WT_RAW + 15) // 16) * 16  # 80


def make_blob(W_int8, rq_m0, rq_sh, rq_bias, oc_block, mt, nz_map_mt):
    """Pack one (oc_block, M-block mt) weight blob = WT_BLK=80 bytes.

    W_int8    : (Cout, Cin) int8
    oc_block  : which group of 4 output channels
    mt        : M-block index within Cin (0..N_IC_BLKS-1)
    nz_map_mt : [N_CFG] int -- NZ channel indices within this M-block
    """
    blob = bytearray()

    # NZ entries: N_CFG x 17B for this M-block
    for slot in range(N_CFG):
        idx    = int(nz_map_mt[slot])
        ic_abs = mt * 16 + idx
        entry  = bytearray([idx & 0xFF])
        for spec in range(4):
            for grp in range(4):
                oc = oc_block * 4 + grp
                if spec == 0 and oc < W_int8.shape[0]:
                    entry.append(int(W_int8[oc, ic_abs]) & 0xFF)
                else:
                    entry.append(0)
        assert len(entry) == 17
        blob += entry

    # Requant: exactly POC_SIZE=4 entries (kernel only reads 4 in poc=0 mode)
    for local_ch in range(POC_SIZE):
        oc = oc_block * 4 + local_ch
        if oc < W_int8.shape[0]:
            m0 = int(rq_m0[oc]) & 0xFFFF
            sh = int(rq_sh[oc]) & 0x1F
            bi = int(rq_bias[oc])
        else:
            m0, sh, bi = 0x4000, 14, 0
        blob += bytes([m0 & 0xFF, (m0 >> 8) & 0xFF, sh,
                       bi & 0xFF, (bi >> 8) & 0xFF,
                       (bi >> 16) & 0xFF, (bi >> 24) & 0xFF])

    assert len(blob) == _WT_RAW, f"blob len {len(blob)} != {_WT_RAW}"
    blob += bytes(WT_BLK - len(blob))
    return bytes(blob)


def nspec_sizes(NS):
    """WT blob sizes for the all-NS-SPEC (B-w) mode."""
    ock = NS * GROUPS                       # output channels per kick
    esz = 1 + ock                           # NZ entry bytes (idx + NS*4 weights)
    raw = N_CFG * esz + ock * 7             # N_CFG entries + ock requant
    blk = ((raw + 15) // 16) * 16           # 16B-aligned
    return ock, esz, raw, blk


def make_blob_nspec(W_int8, rq_m0, rq_sh, rq_bias, oc_block, mt, nz_map_mt, NS):
    """B-w: pack one (oc_block, M-block) blob filling ALL NS SPEC (output-parallel).

    SPEC s, group g → oc = oc_block*(NS*4) + s*4 + g. All NS SPEC share the same ACT M-block
    (broadcast in i_spec); each SPEC computes 4 different output channels → NS*4 OC/kick.
    Weight byte order = local(spec*4+grp) → matches RTL wt_rd_arr[b>>2][b&3] (b=byte index).
    Requant order = local → matches out_ch=flush_spec*4+flush_g and rq_load_ch sequential.
    """
    ock, esz, raw, blk = nspec_sizes(NS)
    blob = bytearray()
    for slot in range(N_CFG):
        idx    = int(nz_map_mt[slot])
        ic_abs = mt * 16 + idx
        entry  = bytearray([idx & 0xFF])
        for spec in range(NS):
            for grp in range(GROUPS):
                oc = oc_block * ock + spec * GROUPS + grp
                entry.append(int(W_int8[oc, ic_abs]) & 0xFF if oc < W_int8.shape[0] else 0)
        assert len(entry) == esz
        blob += entry
    for local in range(ock):
        oc = oc_block * ock + local
        if oc < W_int8.shape[0]:
            m0 = int(rq_m0[oc]) & 0xFFFF; sh = int(rq_sh[oc]) & 0x1F; bi = int(rq_bias[oc])
        else:
            m0, sh, bi = 0x4000, 14, 0
        blob += bytes([m0 & 0xFF, (m0 >> 8) & 0xFF, sh,
                       bi & 0xFF, (bi >> 8) & 0xFF, (bi >> 16) & 0xFF, (bi >> 24) & 0xFF])
    assert len(blob) == raw, f"nspec blob {len(blob)} != {raw}"
    blob += bytes(blk - raw)
    return bytes(blob)


def main():
    meta = json.load(open(os.path.join(SPARSE_DIR, 'metadata.json')))
    req  = json.load(open(os.environ.get('REQ_FILE', os.path.join(SPARSE_DIR, 'requant_params.json'))))
    os.makedirs(OUT_DIR, exist_ok=True)

    layer_params = {}
    nz_map_all   = {}

    for lid in range(63):
        k = f'layer_{lid:02d}'
        m = meta[k]
        ws   = m['weight_shape']    # [Cout, Cin, kH, kW]
        Cout, Cin, kH, kW = ws
        isdw = bool(m['is_depthwise'])

        # Only convert conv1x1, Cin%16==0, Cout%4==0, has requant, non-FP32
        if kH != 1 or kW != 1 or isdw:
            continue
        if Cin % 16 != 0:
            continue
        if k not in req:
            continue

        W  = np.fromfile(os.path.join(SPARSE_DIR, 'weights_bin', f'{k}.bin'),
                         dtype=np.int8).reshape(Cout, Cin)
        b  = np.fromfile(os.path.join(SPARSE_DIR, 'bias_bin', f'{k}.bin'),
                         dtype=np.int32)
        m0 = np.array(req[k]['M0'],    dtype=np.uint16)
        sh = np.array(req[k]['shift'], dtype=np.uint8)
        # 2026-07-07: cho phép Cout không chia hết 4 (L60/61 Cout=21 — detect head INT8):
        # pad W/bias/M0/shift lên bội 4 bằng 0 (kênh pad → psum=0, requant M0=0 → out 0,
        # host crop [:Cout] như mọi layer — pad-OC=0 mechanism có sẵn ở block cuối 48).
        if Cout % 4 != 0:
            pad = 4 - (Cout % 4)
            W  = np.vstack([W,  np.zeros((pad, Cin), np.int8)])
            b  = np.concatenate([b,  np.zeros(pad, np.int32)])
            m0 = np.concatenate([m0, np.zeros(pad, np.uint16)])
            sh = np.concatenate([sh, np.zeros(pad, np.uint8)])
            Cout_real, Cout = Cout, Cout + pad
        else:
            Cout_real = Cout

        N_IC_BLKS = Cin  // 16
        if NCFG_AUTO:
            global N_CFG
            _nzcol = (W.reshape(Cout, N_IC_BLKS, 16) != 0).any(axis=0)   # [N_IC,16]
            N_CFG = max(1, int(_nzcol.sum(axis=1).max()))
        # B-w: OC per kick = NSPEC*4 (all-SPEC) or 4 (legacy SPEC0-only)
        if NSPEC > 0:
            ock, esz, raw, blk = nspec_sizes(NSPEC)
            N_OCB = (Cout + ock - 1) // ock     # ceil(Cout/(NSPEC*4)); last block zero-padded
            cur_blk = blk
        else:
            N_OCB   = Cout // 4
            cur_blk = WT_BLK

        # NZ selection: top N_CFG per M-block by summed |w| over all output channels
        nz_map = np.zeros((N_IC_BLKS, N_CFG), dtype=np.int32)
        for mt in range(N_IC_BLKS):
            w_mblk = W[:, mt*16:(mt+1)*16]           # (Cout, 16)
            score  = np.sum(np.abs(w_mblk), axis=0)  # (16,)
            top    = np.argsort(-score)[:N_CFG]
            top.sort()
            nz_map[mt] = top

        # Write N_OCB x N_IC_BLKS blobs: [ob=0,mt=0]..[ob=0,mt=N_ICB-1],[ob=1,mt=0]..
        # DDR index = ob*N_ICB + mt  =>  layer_seq wt_load_addr matches exactly
        out_path = os.path.join(OUT_DIR, f'{k}.bin')
        with open(out_path, 'wb') as f:
            for ob in range(N_OCB):
                for mt in range(N_IC_BLKS):
                    if NSPEC > 0:
                        blob = make_blob_nspec(W, m0, sh, b, ob, mt, nz_map[mt], NSPEC)
                    else:
                        blob = make_blob(W, m0, sh, b, ob, mt, nz_map[mt])
                    f.write(blob)

        wt_total = N_OCB * N_IC_BLKS * cur_blk
        layer_params[k] = {
            'Cin':       Cin,
            'Cout':      Cout_real,   # host crop [:Cout] = kênh THẬT (pad 21→24 chỉ trong blob)
            'N_IC_BLKS': N_IC_BLKS,
            'N_OCB':     N_OCB,
            'N_CFG':     N_CFG,
            'NSPEC':     NSPEC if NSPEC > 0 else 1,       # SPEC filled (1=legacy SPEC0-only)
            'OC_PER_KICK': (NSPEC*4 if NSPEC > 0 else 4), # = REG_COUT for inference
            'WT_BLK':    cur_blk,
            'wt_total':  wt_total,
        }
        nz_map_all[k] = nz_map.tolist()
        print(f'{k}: Cin={Cin} Cout={Cout} N_ICB={N_IC_BLKS} N_OCB={N_OCB} '
              f'NSPEC={NSPEC if NSPEC>0 else 1} OCK={NSPEC*4 if NSPEC>0 else 4} '
              f'WT_BLK={cur_blk} total={wt_total}B  -> {out_path}')

    json.dump(layer_params, open(os.path.join(OUT_DIR, 'layer_params.json'), 'w'), indent=2)
    json.dump(nz_map_all,   open(os.path.join(OUT_DIR, 'nz_map.json'),       'w'))
    total_all = sum(p['wt_total'] for p in layer_params.values())
    print(f'\nDone. {len(layer_params)} conv1x1 layers -> {OUT_DIR}')
    print(f'Total weight size: {total_all}B ({total_all/1024/1024:.2f}MB)')
    if NSPEC > 0:
        ock, esz, raw, blk = nspec_sizes(NSPEC)
        print(f'B-w mode NSPEC={NSPEC}: OC/kick={ock}, entry={esz}B, WT_BLK={blk}B. '
              f'⚠️ Set cnn_defines N_SPEC_ACT={NSPEC} + inference REG_COUT={ock} + rebuild.')
    else:
        print(f'LEGACY SPEC0-only: WT_BLK={WT_BLK}B (N_CFG={N_CFG}, poc_size={POC_SIZE}). Set NSPEC=<N> env for B-w.')


if __name__ == '__main__':
    main()
