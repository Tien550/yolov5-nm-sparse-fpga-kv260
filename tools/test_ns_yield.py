#!/usr/bin/env python3
"""test_ns_yield.py — Bring-up netseq_v2 Rev172 trên board (SAU khi nạp bit build172).

3 bước gate, mỗi bước so BIT-EXACT với đường chạy thường (per-layer kick):
  B1  NS regression : chuỗi L26→L30 chạy 1 segment NS (như Rev144) == 2 kick rời?
  B2  YIELD/resume  : desc0(L26, w30.bit1=YIELD) → done → host "CPU-op" → kick lại
                      → desc1(L30, LAST) == 2 kick rời?  (kiểm resume_li giữ đúng chỗ)
  B3  (in dbg)      : ns_err / dbg state sau mỗi bước.

Chạy trên board:  cd ~/rtl_bd && sudo -E env C3FAST=1 HEADQ=1 \
    SPARSE_EXTRACTED=/home/ubuntu/yolov5/runs/quant/extracted_c3 \
    UNI_NAME=weights_unified_c3_ns12c python3 test_ns_yield.py
(KHÔNG bật NS/INTWAIT env — script tự điều khiển từng kick, busy-poll cho chắc.)
"""
import os, sys, time, struct, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_unified_v4 as v4
from inference_unified_v4 import (HWAccel, preload_weights, conv1x1_hw,
                                  DESC_OFF, CH_A_OFF, ACT_OFF,
                                  REG_IN_ADDR_LO, REG_LAYER_MODE, REG_CTRL,
                                  _LAYER_MODE_VAL, _LM_NS, AP_START, AP_DONE)
import inference_cpu as C

MAGIC = 0x4B44
FLAG_LAST, FLAG_YIELD = 0x1, 0x2


def build_c1x1_desc(hw, x_off, out_off, k, p, wt_offsets, relu, n_ic_eff, N_POS, lm):
    """Desc 32-word cho 1 lớp conv1x1 (ảnh reg-file 0x040..0x1E0) — theo _conv1x1_ns."""
    OCK = p.get('OC_PER_KICK', 4)
    d = [0] * 32
    d[0] = 16;         d[1] = OCK
    d[4] = p['N_CFG']; d[6] = 2;   d[7] = 1 if relu else 0
    wt = hw.phys_base + wt_offsets[k]; d[8]  = wt & 0xFFFFFFFF; d[9]  = wt >> 32
    ia = hw.phys_base + x_off;         d[10] = ia & 0xFFFFFFFF; d[11] = ia >> 32
    oa = hw.phys_base + out_off;       d[12] = oa & 0xFFFFFFFF; d[13] = oa >> 32
    d[14] = 1;  d[16] = 0;  d[17] = OCK * 16
    d[20] = N_POS * n_ic_eff
    d[21] = lm
    d[22] = p['N_OCB']; d[23] = N_POS; d[24] = n_ic_eff
    d[25] = p['WT_BLK']; d[26] = 256
    d[31] = MAGIC
    return d


def pack_act(x, Cin, N_IC, N_POS):
    HW_ = x.shape[1] * x.shape[2]
    P_pad = N_POS * 16
    buf = np.zeros((N_IC * 16, P_pad), np.int8)
    buf[:Cin, :HW_] = x.reshape(Cin, HW_)
    return buf.reshape(N_IC, 16, N_POS, 16).transpose(2, 0, 1, 3).reshape(-1).tobytes()


def detile(raw, N_OCB, N_POS, OCK, Cout, H, W):
    r = np.ascontiguousarray(
        np.frombuffer(raw, np.int8).reshape(N_OCB, N_POS, OCK, 16).transpose(0, 2, 1, 3)
    ).reshape(N_OCB * OCK, N_POS * 16)[:Cout]
    return r[:, :H * W].reshape(Cout, H, W)


def run_segment(hw, table_bytes, n_kicks):
    """Nạp desc table + kick n_kicks lần (YIELD: mỗi kick chạy tới yield/last kế tiếp)."""
    hw.dma_write(DESC_OFF, table_bytes)
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + DESC_OFF)
    hw.wr(REG_LAYER_MODE, _LAYER_MODE_VAL | _LM_NS)
    for i in range(n_kicks):
        hw.wr(REG_CTRL, AP_START)
        t0 = time.time()
        while not (hw.regs[REG_CTRL >> 2] & AP_DONE):
            if time.time() - t0 > 20: raise RuntimeError(f'NS TIMEOUT (kick {i})')
        # host "CPU-op" giả giữa các đoạn (chứng minh yield trả quyền thật)
        time.sleep(0.002)
    hw.wr(REG_LAYER_MODE, 0)   # rời NS mode


def main():
    C.E = v4.SPARSE_DIR
    meta, req, weights, bias = C.load_model()
    layer_params = json.load(open(os.path.join(v4.UNI_DIR, 'layer_params.json')))
    hw = HWAccel()
    wt_offsets = preload_weights(hw, layer_params)

    # ── chuỗi thử: L26 (512→512, 13²) → L30 (512→256, 13²) — cùng cặp PoC Rev144 ──
    k26, k30 = 'layer_26', 'layer_30'
    p26, p30 = layer_params[k26], layer_params[k30]
    rng = np.random.default_rng(7)
    act = rng.integers(-40, 40, (p26['Cin'], 13, 13), dtype=np.int8)

    PROF = {}
    # GOLDEN: 2 kick rời qua đường chuẩn (đã board-proven)
    g26 = conv1x1_hw(hw, act, k26, layer_params, wt_offsets, True, PROF)
    g30 = conv1x1_hw(hw, np.asarray(g26), k30, layer_params, wt_offsets, True, PROF)
    print(f'[golden] L30 out {g30.shape} sum={int(np.asarray(g30).astype(np.int64).sum())}')

    # chuẩn bị input tiled + vùng out cho đường NS (arena riêng, không đè golden)
    H = W = 13; HW_ = H * W; N_POS = (HW_ + 15) // 16
    a_off  = CH_A_OFF
    o26_off = CH_A_OFF + 0x100000
    o30_off = CH_A_OFF + 0x200000
    act_b = pack_act(act, p26['Cin'], p26['N_IC_BLKS'], N_POS)
    lm = _LAYER_MODE_VAL | _LM_NS

    OCK26 = p26['OC_PER_KICK']; cin30_pad = p26['N_OCB'] * OCK26   # pad-OC=0 → đọc cả chunk pad
    d26 = build_c1x1_desc(hw, a_off, o26_off, k26, p26, wt_offsets, True,
                          p26['N_IC_BLKS'], N_POS, lm)
    d30 = build_c1x1_desc(hw, o26_off, o30_off, k30, p30, wt_offsets, True,
                          cin30_pad // 16, N_POS, lm)
    # ⚠️ chain trực tiếp DDR: L26 OUT (oc-major) ≠ ACT chunk format → như Rev144 cần OT.
    # Đơn giản + đủ cho gate netseq: L30 đọc lại từ act CHUẨN do host pack (không chain DDR).
    a30_off = CH_A_OFF + 0x300000

    def stage_l30_input():
        raw = hw.dma_read(o26_off, p26['N_OCB'] * N_POS * OCK26 * 16)
        x26 = detile(raw, p26['N_OCB'], N_POS, OCK26, p26['Cout'], H, W)
        hw.dma_write(a30_off, pack_act(x26, p30['Cin'], p30['N_IC_BLKS'], N_POS))
        return x26

    d30b = build_c1x1_desc(hw, a30_off, o30_off, k30, p30, wt_offsets, True,
                           p30['N_IC_BLKS'], N_POS, lm)

    def read_o30():
        raw = hw.dma_read(o30_off, p30['N_OCB'] * N_POS * p30['OC_PER_KICK'] * 16)
        return detile(raw, p30['N_OCB'], N_POS, p30['OC_PER_KICK'], p30['Cout'], H, W)

    def cmp(tag, out):
        d = np.abs(np.asarray(g30).astype(np.int16) - out.astype(np.int16))
        n = int((d > 0).sum())
        print(f'[{tag}] mismatch={n}/{out.size} max={int(d.max())} '
              + ('*** PASS ***' if n == 0 else '*** FAIL ***'))
        return n == 0

    hw.dma_write(a_off, act_b)

    # ══ B1: NS regression — segment 2 desc, desc1 dùng input host-staged ══
    # kick 1: chạy d26 (YIELD) → host stage input L30 → kick 2: d30b (LAST)
    # (đây vừa là B1 vừa là B2 gộp: yield giữa 2 lớp = đúng pattern DW-breakpoint thật)
    t26 = d26[:]; t26[30] = FLAG_YIELD
    t30 = d30b[:]; t30[30] = FLAG_LAST
    table = b''.join(struct.pack('<32I', *[w & 0xFFFFFFFF for w in d]) for d in (t26, t30))
    hw.dma_write(o30_off, bytes(p30['N_OCB'] * N_POS * p30['OC_PER_KICK'] * 16))  # clear

    hw.dma_write(DESC_OFF, table)
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + DESC_OFF)
    hw.wr(REG_LAYER_MODE, _LAYER_MODE_VAL | _LM_NS)
    # kick 1 → yield sau L26
    hw.wr(REG_CTRL, AP_START); t0 = time.time()
    while not (hw.regs[REG_CTRL >> 2] & AP_DONE):
        if time.time() - t0 > 20: raise RuntimeError('TIMEOUT kick1 (yield)')
    x26 = stage_l30_input()                      # host CPU-op thật giữa yield
    d1 = np.abs(np.asarray(g26).astype(np.int16) - x26.astype(np.int16))
    print(f'[yield-L26] mismatch={int((d1>0).sum())}/{x26.size} max={int(d1.max())}')
    # ⚠️ BUG BẮT ĐƯỢC 2026-07-10: desc0 GHI ĐÈ REG_IN_ADDR (w10 = act addr L26) mà netseq
    # resume latch desc_base từ CHÍNH reg đó → fetch "desc" từ vùng act → magic sai → ns_err
    # → done rỗng. FIX host: ghi lại desc_base trước MỌI kick resume. (RTL fix cho build sau:
    # dbase <= resume_pend ? dbase : desc_base.)
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + DESC_OFF)
    # kick 2 → resume desc1 → LAST
    hw.wr(REG_CTRL, AP_START); t0 = time.time()
    while not (hw.regs[REG_CTRL >> 2] & AP_DONE):
        if time.time() - t0 > 20: raise RuntimeError('TIMEOUT kick2 (resume)')
    hw.wr(REG_LAYER_MODE, 0)
    ok = cmp('B1+B2 NS yield/resume', read_o30())

    print('\n=== TEST_NS_YIELD ' + ('ALL PASS' if ok else 'FAILED') + ' ===')


if __name__ == '__main__':
    main()
