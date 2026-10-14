#!/usr/bin/env python3
"""debug_gate_d.py — Chẩn đoán Gate D fail bằng LADDER + HYPOTHESIS MATCHING.

T1: 1 desc  L17 ACTL@0 → DDR (kiểm desc/NS cơ bản)
T2: 2 desc  L17→DDR, L18 resident@0 (ACT_TOTAL=0) → DDR (kiểm resident-in @0)
T4: 2 desc  L17(SCATOUT→A) → L20(RESINBASE=A) → DDR (cặp fail của Gate D)

Nếu T4 fail: dựng act_uram ẢO trên host, mô phỏng scatter theo TỪNG GIẢ THUYẾT bug,
đọc lại theo cách consumer đọc (tiled chuẩn + resin) → mc1_hyp; giả thuyết nào khớp
bit-exact với mc1_ns đo được = thủ phạm.
  H0: scatter ĐÚNG thiết kế (pt*cpad + ob*48 + oc, clamp ob cuối)
  H1: KHÔNG có ob offset (mọi ob đè word pt*cpad+oc)
  H2: KHÔNG clamp (ob cuối ghi cả 48, pad=0 tràn sang tile kế)
  H3: RESINBASE không áp (consumer đọc @0 = x6)
  H6: rơi CHW mode (base=pos+scat_base, stride=npos)
"""
import os, sys, time, struct, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_unified_v4 as v4
from inference_unified_v4 import (HWAccel, preload_weights, conv1x1_hw,
                                  DESC_OFF, CH_A_OFF, REG_IN_ADDR_LO, REG_LAYER_MODE,
                                  REG_CTRL, _LAYER_MODE_VAL, AP_START, AP_DONE)
from cpu_conv import quant_conv
import inference_cpu as C

MAGIC = 0x4B44
FLAG_LAST, FLAG_YIELD = 0x1, 0x2
LM_BASE, LM_SCAT, LM_RESIN = 0x803, 0x803 | 0x2000 | 0x10000, 0x803 | 0x20000
NPOS = 43
R_A = 0x2B000
D_X, D_O1, D_O2 = CH_A_OFF, CH_A_OFF + 0x100000, CH_A_OFF + 0x200000


def pack_act(x, Cin, N_IC):
    HW_ = x.shape[1] * x.shape[2]
    buf = np.zeros((N_IC * 16, NPOS * 16), np.int8)
    buf[:Cin, :HW_] = x.reshape(Cin, HW_)
    return buf.reshape(N_IC, 16, NPOS, 16).transpose(2, 0, 1, 3).reshape(-1).tobytes()


def detile(raw, N_OCB, OCK, Cout, H=26, W=26):
    r = np.ascontiguousarray(
        np.frombuffer(raw, np.int8).reshape(N_OCB, NPOS, OCK, 16).transpose(0, 2, 1, 3)
    ).reshape(N_OCB * OCK, NPOS * 16)[:Cout]
    return r[:, :H * W].reshape(Cout, H, W)


def conv_desc(hw, p, wt_off, lm, in_addr, out_addr, cpad=0, act_total=None, flags=0):
    OCK = p['OC_PER_KICK']
    d = [0] * 32
    d[0] = 16; d[1] = OCK; d[4] = p['N_CFG']; d[6] = 2; d[7] = 1
    w = hw.phys_base + wt_off; d[8] = w & 0xFFFFFFFF; d[9] = w >> 32
    d[10] = in_addr & 0xFFFFFFFF; d[11] = in_addr >> 32
    d[12] = out_addr & 0xFFFFFFFF; d[13] = out_addr >> 32
    d[14] = 1; d[17] = OCK * 16; d[19] = cpad
    d[20] = (NPOS * p['N_IC_BLKS']) if act_total is None else act_total
    d[21] = lm; d[22] = p['N_OCB']; d[23] = NPOS; d[24] = p['N_IC_BLKS']
    d[25] = p['WT_BLK']; d[26] = 256; d[30] = flags; d[31] = MAGIC
    return d


def run_seg(hw, descs):
    table = b''.join(struct.pack('<32I', *[x & 0xFFFFFFFF for x in d]) for d in descs)
    hw.dma_write(DESC_OFF, table)
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + DESC_OFF)
    hw.wr(REG_LAYER_MODE, _LAYER_MODE_VAL | 0x800)
    hw.wr(REG_CTRL, AP_START); t0 = time.time()
    while not (hw.regs[REG_CTRL >> 2] & AP_DONE):
        if time.time() - t0 > 30: raise RuntimeError('NS TIMEOUT')
    hw.wr(REG_LAYER_MODE, 0)


def cmp(tag, a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    n = int((d > 0).sum())
    print(f'  [{tag}] mismatch={n}/{a.size} max={int(d.max())} ' + ('PASS' if n == 0 else 'FAIL'))
    return n == 0


# ── mô phỏng scatter + đọc lại (act_uram ảo) ────────────────────────────────
def sim_scatter_read(cv1_tiled_words, hyp, npos=NPOS, cpad=128, ock=48, nocb=3,
                     base_w=R_A >> 4, init=None):
    """cv1_tiled_words[ob][pt] = list 48 word (mỗi word 16 byte) engine nhả ra out_bram.
    Trả act_uram ảo (mảng word) sau scatter theo giả thuyết hyp."""
    au = dict(init or {})
    for ob in range(nocb):
        for pt in range(npos):
            words = cv1_tiled_words[ob][pt]     # 48 word × 16B
            if hyp == 'H0':
                ockw = min(ock, cpad - ob * ock)
                for oc in range(ockw):
                    au[base_w + pt * cpad + ob * ock + oc] = words[oc]
            elif hyp == 'H1':
                for oc in range(ock):
                    au[base_w + pt * cpad + oc] = words[oc]
            elif hyp == 'H2':
                for oc in range(ock):
                    au[base_w + pt * cpad + ob * ock + oc] = words[oc]
            elif hyp == 'H6':
                for oc in range(ock):
                    au[base_w + pt + oc * npos] = words[oc]
    return au


def read_consumer(au, base_w, n_ic, npos=NPOS):
    """Đọc như consumer conv1x1 BI + resin: chunk (pos,ic) @ (pos*n_ic+ic)*16 word + base."""
    out = np.zeros((n_ic * 16, npos * 16), np.int8)
    for pos in range(npos):
        for ic in range(n_ic):
            for woff in range(16):
                w = au.get(base_w + (pos * n_ic + ic) * 16 + woff)
                if w is not None:
                    out[ic * 16 + woff, pos * 16:(pos + 1) * 16] = np.frombuffer(w, np.int8)
    return out


def main():
    C.E = v4.SPARSE_DIR
    meta, req, weights, bias = C.load_model()
    lp = json.load(open(os.path.join(v4.UNI_DIR, 'layer_params.json')))
    hw = HWAccel()
    wt = preload_weights(hw, lp)
    p17, p18, p20 = lp['layer_17'], lp['layer_18'], lp['layer_20']

    rng = np.random.default_rng(19)
    x6 = rng.integers(-40, 40, (256, 26, 26), dtype=np.int8)
    PROF = {}
    cv1 = np.asarray(conv1x1_hw(hw, x6, 'layer_17', lp, wt, True, PROF))
    cv2 = np.asarray(conv1x1_hw(hw, x6, 'layer_18', lp, wt, True, PROF))
    mc1 = np.asarray(conv1x1_hw(hw, cv1, 'layer_20', lp, wt, True, PROF))

    hw.dma_write(D_X, pack_act(x6, 256, 16))

    # T1
    run_seg(hw, [conv_desc(hw, p17, wt['layer_17'], LM_BASE, hw.phys_base + D_X,
                           hw.phys_base + D_O1, flags=FLAG_LAST)])
    o = detile(hw.dma_read(D_O1, p17['N_OCB'] * NPOS * p17['OC_PER_KICK'] * 16),
               p17['N_OCB'], p17['OC_PER_KICK'], 128)
    t1 = cmp('T1 desc cơ bản (L17→DDR)', cv1, o)

    # T2
    run_seg(hw, [conv_desc(hw, p17, wt['layer_17'], LM_BASE, hw.phys_base + D_X,
                           hw.phys_base + D_O1),
                 conv_desc(hw, p18, wt['layer_18'], LM_BASE, 0, hw.phys_base + D_O2,
                           act_total=0, flags=FLAG_LAST)])
    o2 = detile(hw.dma_read(D_O2, p18['N_OCB'] * NPOS * p18['OC_PER_KICK'] * 16),
                p18['N_OCB'], p18['OC_PER_KICK'], 128)
    t2 = cmp('T2 resident-in @0 (L18)', cv2, o2)

    # T4
    run_seg(hw, [conv_desc(hw, p17, wt['layer_17'], LM_SCAT, hw.phys_base + D_X, R_A, cpad=128),
                 conv_desc(hw, p20, wt['layer_20'], LM_RESIN, R_A, hw.phys_base + D_O1,
                           act_total=0, flags=FLAG_LAST)])
    mc1_ns = detile(hw.dma_read(D_O1, p20['N_OCB'] * NPOS * p20['OC_PER_KICK'] * 16),
                    p20['N_OCB'], p20['OC_PER_KICK'], 128)
    t4 = cmp('T4 scatter→A + RESINBASE (L20)', mc1, mc1_ns)

    if t4:
        print('=== T4 PASS — lỗi Gate D nằm ở desc sau (L22/graphop/L19) ===')
        return

    # ── T4 FAIL → hypothesis matching ──
    # dựng out_bram words của L17 từ cv1 golden: kick (ob,pt) nhả 48 word; word oc = 16 sp
    # của kênh ob*48+oc (kênh >=128 → pad: engine requant m0=0 → 0)
    cv1p = np.zeros((3 * 48, NPOS * 16), np.int8)
    cv1p[:128, :676] = cv1.reshape(128, 676)
    words = [[[cv1p[ob * 48 + oc, pt * 16:(pt + 1) * 16].tobytes() for oc in range(48)]
              for pt in range(NPOS)] for ob in range(3)]

    w20 = weights['layer_20'].reshape(128, 128)
    m20 = np.array(req['layer_20']['M0'], np.uint16)
    s20 = np.array(req['layer_20']['shift'], np.uint8)
    b20 = bias['layer_20']

    for hyp in ('H0', 'H1', 'H2', 'H6'):
        au = sim_scatter_read(words, hyp)
        a_eff = read_consumer(au, R_A >> 4, 8)[:, :676].reshape(128, 26, 26)
        mc1_h = quant_conv(a_eff, weights['layer_20'], b20, m20, s20,
                           stride=1, pad=0, is_depthwise=False, apply_relu=True)
        d = np.abs(mc1_h.astype(np.int16) - mc1_ns.astype(np.int16))
        print(f'  [HYP {hyp}] mismatch={int((d>0).sum())}/{d.size} max={int(d.max())}'
              + ('   ← ★ KHỚP — thủ phạm!' if (d > 0).sum() == 0 else ''))
    # H3: RESINBASE không áp — consumer đọc @0 nơi x6 vẫn nằm (stride 8 chunk)
    x6b = np.frombuffer(pack_act(x6, 256, 16), np.int8).copy()
    au0 = {i: x6b[i * 16:(i + 1) * 16].tobytes() for i in range(len(x6b) // 16)}
    a_eff = read_consumer(au0, 0, 8)[:, :676].reshape(128, 26, 26)
    mc1_h = quant_conv(a_eff, weights['layer_20'], b20, m20, s20,
                       stride=1, pad=0, is_depthwise=False, apply_relu=True)
    d = np.abs(mc1_h.astype(np.int16) - mc1_ns.astype(np.int16))
    print(f'  [HYP H3 resin-không-áp] mismatch={int((d>0).sum())}/{d.size} max={int(d.max())}'
          + ('   ← ★ KHỚP — thủ phạm!' if (d > 0).sum() == 0 else ''))


if __name__ == '__main__':
    main()
