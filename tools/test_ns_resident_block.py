#!/usr/bin/env python3
"""test_ns_resident_block.py — GATE D (Rev173): block resident + graphop IN-CHAIN trên board.

Chuỗi Shuffle Block 3 (26², block nhỏ nhất fit trọn act_uram 1.5MB) chạy MỘT segment netseq
với 1 YIELD (DW trên CPU), mọi transition trung gian KHÔNG rời chip:

  desc0 L17 cv1 (256→128): ACTL x6→act_uram@0, out SCATTER TILED → vùng A   [RESOUT|SCATOUT]
  desc1 L18 cv2 (256→128): x6 resident @0 (ACT_TOTAL=0),        → vùng C   [RESOUT|SCATOUT]
  desc2 L20 mc1 (128→128): input RESIDENT @A [RESINBASE], out DDR + YIELD
  ---- host: detile mc1 → dw_cpu_fast (DW21) → pack mdw → DDR ----
  desc3 L22 mc2 (128→128): ACTL mdw→act_uram@0 (đè x6 đã dùng), → vùng B   [RESOUT|SCATOUT]
  desc4 GRAPHOP LT=4: resid_cat(A,B)‖cat C → vùng DST (rescale on-chip, TILED in/out)
  desc5 L19 (256→256): input RESIDENT @DST [RESINBASE], out DDR + LAST

So BIT-EXACT với đường chuẩn per-layer (conv1x1_hw + dw_cpu_fast + resid_cat numpy).
Đây là bằng chứng silicon cho: multi-ob tiled scatter (L17: N_OCB=3, clamp ob cuối 32ch),
RESINBASE multi-region, graphop end-to-end, YIELD giữa chuỗi resident.

Chạy trên board (bit rev173):
  cd ~/rtl_bd && sudo -E env SPARSE_EXTRACTED=/home/ubuntu/yolov5/runs/quant/extracted_c3 \
      UNI_NAME=weights_unified_c3_ns12c python3 test_ns_resident_block.py
"""
import os, sys, time, struct, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_unified_v4 as v4
from inference_unified_v4 import (HWAccel, preload_weights, conv1x1_hw, dw_cpu_fast,
                                  DESC_OFF, CH_A_OFF,
                                  REG_IN_ADDR_LO, REG_LAYER_MODE, REG_CTRL,
                                  _LAYER_MODE_VAL, AP_START, AP_DONE)
import inference_cpu as C

MAGIC = 0x4B44
FLAG_LAST, FLAG_YIELD = 0x1, 0x2
LM_BASE   = 0x1 | 0x2 | 0x800          # lseq | BI | NS
LM_SCAT   = LM_BASE | 0x2000 | 0x10000 # + RESOUT + SCATOUT (producer → act_uram tiled)
LM_RESIN  = LM_BASE | 0x20000          # + RESINBASE (consumer đọc act_uram @in_addr)
LM_GOP    = 0x1 | 0x800                # graphop: lseq | NS (không BI)

# ── bản đồ vùng act_uram (byte, đều chia hết 256) — block 3 @26²: N_POS=43 tiles ──
NPOS   = 43                       # ceil(676/16)
R_X    = 0x00000                  # x6 input 256ch (ACTL @0): 43*16*256 = 0x2B000
R_A    = 0x2B000                  # cv1 128ch: 43*8*256 = 0x15800
R_B    = 0x40800                  # mc2 128ch
R_C    = 0x56000                  # cv2 128ch
R_DST  = 0x6B800                  # graphop out 256ch: 0x2B000 → end 0x96800 < 1.5MB ✓
# DDR offsets (trong arena A của v4 — không đụng ACT/OUT/WT chuẩn)
D_X    = CH_A_OFF                 # x6 packed tiled
D_MC1  = CH_A_OFF + 0x100000      # L20 out (DDR, cho DW)
D_MDW  = CH_A_OFF + 0x200000      # mdw packed (input L22)
D_FIN  = CH_A_OFF + 0x300000      # L19 out cuối


def find_best_shift(M):
    if M <= 0: return 0, 0
    sh, v = 0, M
    while v < (1 << 15) and sh < 31: v *= 2; sh += 1
    while v >= (1 << 16) and sh > 0: v /= 2; sh -= 1
    m0 = int(round(v))
    return (min(m0, (1 << 16) - 1), sh)


def pack_act(x, Cin, N_IC):
    HW_ = x.shape[1] * x.shape[2]; P_pad = NPOS * 16
    buf = np.zeros((N_IC * 16, P_pad), np.int8)
    buf[:Cin, :HW_] = x.reshape(Cin, HW_)
    return buf.reshape(N_IC, 16, NPOS, 16).transpose(2, 0, 1, 3).reshape(-1).tobytes()


def detile(raw, N_OCB, OCK, Cout, H, W):
    r = np.ascontiguousarray(
        np.frombuffer(raw, np.int8).reshape(N_OCB, NPOS, OCK, 16).transpose(0, 2, 1, 3)
    ).reshape(N_OCB * OCK, NPOS * 16)[:Cout]
    return r[:, :H * W].reshape(Cout, H, W)


def conv_desc(hw, p, wt_off, lm, in_addr, out_addr, cpad=0, act_total=None, flags=0):
    """Desc conv1x1. in_addr/out_addr: DDR phys (thường) hoặc act_uram offset (RESINBASE/SCATOUT).
    act_total=0 = input resident (skip ACTL). cpad = C3_CFG[9:0] cho tiled scatter."""
    OCK = p['OC_PER_KICK']
    d = [0] * 32
    d[0] = 16; d[1] = OCK; d[4] = p['N_CFG']; d[6] = 2; d[7] = 1
    wt = hw.phys_base + wt_off; d[8] = wt & 0xFFFFFFFF; d[9] = wt >> 32
    d[10] = in_addr & 0xFFFFFFFF; d[11] = in_addr >> 32
    d[12] = out_addr & 0xFFFFFFFF; d[13] = out_addr >> 32
    d[14] = 1; d[16] = 0; d[17] = OCK * 16
    d[19] = cpad
    d[20] = (NPOS * p['N_IC_BLKS']) if act_total is None else act_total
    d[21] = lm
    d[22] = p['N_OCB']; d[23] = NPOS; d[24] = p['N_IC_BLKS']
    d[25] = p['WT_BLK']; d[26] = 256
    d[30] = flags; d[31] = MAGIC
    return d


def gop_desc(a, b, c, dst, n_pos16, ch_ab, ch_c, ma, sa, mb, sb, mc, sc, flags=0):
    """Desc graphop LT=4 (register tái dùng: xem graphop_engine.v)."""
    d = [0] * 32
    d[0] = (sa << 16) | ma          # CIN  = {sh_a, m0_a}
    d[1] = (sb << 16) | mb          # COUT = {sh_b, m0_b}
    d[2] = (sc << 16) | mc          # HIN  = {sh_c, m0_c}
    d[3] = n_pos16                  # WIN  = n_pos (bội 16)
    d[8] = b                        # WT   = addr_b (mc2 / B)
    d[10] = a                       # IN   = addr_a (cv1 / A)
    d[12] = c                       # OUT  = addr_c (cv2 / C)
    d[14] = dst                     # STRIDE = addr_dst
    d[15] = 1                       # PAD  = resid_mode
    d[16] = 4                       # LAYER_TYPE = GRAPHOP
    d[21] = LM_GOP
    d[22] = ch_ab; d[24] = ch_c    # N_OCB = ch_ab ; N_ICB = ch_c
    d[30] = flags; d[31] = MAGIC
    return d


def kick_wait(hw, tag):
    hw.wr(REG_CTRL, AP_START); t0 = time.time()
    while not (hw.regs[REG_CTRL >> 2] & AP_DONE):
        if time.time() - t0 > 30: raise RuntimeError(f'NS TIMEOUT ({tag})')


def main():
    C.E = v4.SPARSE_DIR
    meta, req, weights, bias = C.load_model()
    lp = json.load(open(os.path.join(v4.UNI_DIR, 'layer_params.json')))
    hw = HWAccel()
    wt = preload_weights(hw, lp)
    osc = lambda l: meta[f'layer_{l:02d}']['output_scale']
    isc = lambda l: meta[f'layer_{l:02d}']['input_scale']

    p17, p18, p20, p22, p19 = (lp[f'layer_{l:02d}'] for l in (17, 18, 20, 22, 19))
    k21 = 'layer_21'
    w21 = weights[k21]; b21 = bias[k21]
    m21 = np.array(req[k21]['M0'], np.uint16); s21 = np.array(req[k21]['shift'], np.uint8)

    rng = np.random.default_rng(19)
    x6 = rng.integers(-40, 40, (256, 26, 26), dtype=np.int8)

    # ══ GOLDEN: đường per-layer chuẩn (board-proven) ══
    PROF = {}
    cv1 = np.asarray(conv1x1_hw(hw, x6, 'layer_17', lp, wt, True, PROF))
    cv2 = np.asarray(conv1x1_hw(hw, x6, 'layer_18', lp, wt, True, PROF))
    mc1 = np.asarray(conv1x1_hw(hw, cv1, 'layer_20', lp, wt, True, PROF))
    mdw = dw_cpu_fast(mc1, w21, b21, m21, s21, True)
    mc2 = np.asarray(conv1x1_hw(hw, mdw, 'layer_22', lp, wt, True, PROF))
    rc  = C.resid_cat(cv1, osc(17), mc2, osc(22), cv2, osc(18), isc(19))
    gold = np.asarray(conv1x1_hw(hw, rc, 'layer_19', lp, wt, True, PROF))
    print(f'[golden] L19 out {gold.shape} sum={int(gold.astype(np.int64).sum())}')

    # ══ NS CHAIN: 6 desc, 1 yield ══
    ma, sa = find_best_shift(osc(17) / isc(19))
    mb, sb = find_best_shift(osc(22) / isc(19))
    mc_, sc_ = find_best_shift(osc(18) / isc(19))

    d0 = conv_desc(hw, p17, wt['layer_17'], LM_SCAT, hw.phys_base + D_X, R_A, cpad=128)
    d1 = conv_desc(hw, p18, wt['layer_18'], LM_SCAT, 0, R_C, cpad=128, act_total=0)
    d2 = conv_desc(hw, p20, wt['layer_20'], LM_RESIN, R_A, hw.phys_base + D_MC1,
                   act_total=0, flags=FLAG_YIELD)
    d3 = conv_desc(hw, p22, wt['layer_22'], LM_SCAT, hw.phys_base + D_MDW, R_B, cpad=128)
    d4 = gop_desc(R_A, R_B, R_C, R_DST, NPOS * 16, 128, 128, ma, sa, mb, sb, mc_, sc_)
    d5 = conv_desc(hw, p19, wt['layer_19'], LM_RESIN, R_DST, hw.phys_base + D_FIN,
                   act_total=0, flags=FLAG_LAST)
    table = b''.join(struct.pack('<32I', *[w_ & 0xFFFFFFFF for w_ in d])
                     for d in (d0, d1, d2, d3, d4, d5))

    hw.dma_write(D_X, pack_act(x6, 256, 16))
    hw.dma_write(D_FIN, bytes(p19['N_OCB'] * NPOS * p19['OC_PER_KICK'] * 16))  # clear
    hw.dma_write(DESC_OFF, table)
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + DESC_OFF)
    hw.wr(REG_LAYER_MODE, _LAYER_MODE_VAL | 0x800)

    t0 = time.time()
    kick_wait(hw, 'seg1: L17,L18,L20 → yield')
    # host CPU-op: DW21
    mc1_ns = detile(hw.dma_read(D_MC1, p20['N_OCB'] * NPOS * p20['OC_PER_KICK'] * 16),
                    p20['N_OCB'], p20['OC_PER_KICK'], 128, 26, 26)
    dm = np.abs(mc1.astype(np.int16) - mc1_ns.astype(np.int16))
    print(f'[yield mc1] mismatch={int((dm>0).sum())}/{mc1.size} max={int(dm.max())}')
    mdw_ns = dw_cpu_fast(mc1_ns, w21, b21, m21, s21, True)
    hw.dma_write(D_MDW, pack_act(mdw_ns, 128, 8))
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + DESC_OFF)   # desc_base sạch trước resume
    kick_wait(hw, 'seg2: L22,graphop,L19 → last')
    t_chain = time.time() - t0
    hw.wr(REG_LAYER_MODE, 0)

    out = detile(hw.dma_read(D_FIN, p19['N_OCB'] * NPOS * p19['OC_PER_KICK'] * 16),
                 p19['N_OCB'], p19['OC_PER_KICK'], 256, 26, 26)
    d = np.abs(gold.astype(np.int16) - out.astype(np.int16))
    n = int((d > 0).sum())
    print(f'[GATE D block3] mismatch={n}/{out.size} max={int(d.max())}  chain={t_chain*1e3:.1f}ms')
    print('=== GATE_D ' + ('PASS — resident chain + graphop on-chip BIT-EXACT ===' if n == 0
                           else 'FAIL ==='))


if __name__ == '__main__':
    main()
