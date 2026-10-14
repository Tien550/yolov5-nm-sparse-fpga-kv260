#!/usr/bin/env python3
"""Gate tests: conv3x3 (R1 fast path) + depthwise (swc FSM) on unified_acc_top.

Run on KV260 in /home/ubuntu/rtl_bd/:
  sudo python3 test_swc_gate.py

Both tests use LAYER_MODE=0 (single manual kick).
conv3x3: R1 fast path enabled by REG_PAD[10]=1.
  ACT layout: IN[ic*54 + kh*18 + col]   (n_ic × 3-row × 18-col band)
  WT layout:  WT[(ic*9+tap)*17 + 1 + g] (g=0..3 = output channels 0..3)
DW: swc FSM, n_cfg=1 (dense), poc=0 forced, accu_mode=0 forced.
  ACT layout: IN[ch*tile_chw + kh*win + col]  (4 channels × 3×18 band)
  WT layout:  tap*17 + [idx, W_g0..g3, zeros×12] per tap
"""
import ctypes, mmap, os, sys, time
import numpy as np

if os.geteuid() == 0:
    _us = f"/home/ubuntu/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if os.path.isdir(_us) and _us not in sys.path:
        sys.path.insert(0, _us)

# ── Register map ──────────────────────────────────────────────────────────────
UNIFIED_BASE    = 0xA0000000
REG_CTRL        = 0x000
REG_CIN         = 0x040
REG_COUT        = 0x050
REG_HIN         = 0x060
REG_WIN         = 0x070
REG_N_CONFIG    = 0x080
REG_POC         = 0x090
REG_ACCU_MODE   = 0x0A0
REG_APPLY_RELU  = 0x0B0
REG_WT_ADDR_LO  = 0x0C0
REG_IN_ADDR_LO  = 0x0E0
REG_OUT_ADDR_LO = 0x100
REG_STRIDE      = 0x120
REG_PAD         = 0x130
REG_LAYER_TYPE  = 0x140
REG_OUT_BYTES   = 0x150
REG_WT_BYTES    = 0x160
REG_LAYER_MODE  = 0x190
REG_N_OCB       = 0x1A0
REG_N_POS       = 0x1B0
REG_N_ICB       = 0x1C0
REG_WT_BLK_BYTES = 0x1D0
REG_ACT_CHUNK   = 0x1E0
AP_START = 0x01; AP_DONE = 0x02

DMA_TOTAL = 64 * 1024 * 1024
ACT_OFF   = 0x0000000   # ACT buffer
WT_OFF    = 0x0200000   # WT buffer
OUT_OFF   = 0x0C00000   # OUT buffer


# ── HW helpers ────────────────────────────────────────────────────────────────
def open_hw():
    devmem = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    rmap = mmap.mmap(devmem, 0x10000, mmap.MAP_SHARED,
                     mmap.PROT_READ | mmap.PROT_WRITE, offset=UNIFIED_BASE)
    regs = (ctypes.c_uint32 * (0x10000 // 4)).from_buffer(rmap)
    os.close(devmem)
    with open('/sys/module/dma_region/parameters/phys_base') as f:
        phys_base = int(f.read().strip())
    dfd = os.open('/dev/dma_region', os.O_RDWR | os.O_SYNC)
    dma = mmap.mmap(dfd, DMA_TOTAL, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(dfd)
    print(f'[HW] IP@0x{UNIFIED_BASE:08X}  DMA phys=0x{phys_base:08x}')
    return regs, dma, phys_base

def wr(regs, off, val):
    regs[off >> 2] = int(val) & 0xFFFFFFFF

def wr64(regs, off_lo, addr):
    addr = int(addr)
    regs[off_lo >> 2]          = addr & 0xFFFFFFFF
    regs[(off_lo + 0x10) >> 2] = (addr >> 32) & 0xFFFFFFFF

def dput(dma, offset, data):
    n = len(data)
    ctypes.memmove(ctypes.c_char_p(ctypes.addressof(
        (ctypes.c_char * n).from_buffer(dma, offset))), data, n)

def dget(dma, offset, n):
    return bytes(dma[offset:offset+n])

def wait_done(regs, timeout=10.0):
    t0 = time.time()
    while not (regs[REG_CTRL >> 2] & AP_DONE):
        if time.time() - t0 > timeout:
            raise RuntimeError('TIMEOUT AP_DONE')


# ── Requant helpers ───────────────────────────────────────────────────────────
def pack_rq(m0, sh, bias, n=4):
    """Pack n × 7B requant entries [m0_lo, m0_hi, sh_hw, b0..b3].
    sh is the extra shift (golden_rq convention); hardware rr_unit needs sh+16."""
    m0 = int(m0) & 0xFFFF; sh_hw = (int(sh) + 16) & 0x1F; bi = int(bias)
    e = bytes([m0 & 0xFF, (m0 >> 8) & 0xFF, sh_hw,
               bi & 0xFF, (bi >> 8) & 0xFF, (bi >> 16) & 0xFF, (bi >> 24) & 0xFF])
    return e * n

def golden_rq(acc, m0, sh, bias, relu=False):
    """acc int32 → requantized int8 matching rr_unit formula."""
    m0 = int(m0); sh = int(sh); bi = int(bias)
    v = (acc.astype(np.int64) * m0 + (1 << (sh + 15))) >> (sh + 16)
    v = (v + bi).astype(np.int32)
    if relu:
        v = np.where(v >= 0, v, v >> 3)
    return np.clip(v, -128, 127).astype(np.int8)


# ── TEST 1: conv3x3 via R1 fast path ─────────────────────────────────────────
def test_c3x3_r1(regs, dma, phys_base):
    print('\n=== TEST 1: conv3x3 R1 (n_ic=4, n_oc=4, 16 positions, 1 row) ===')
    N_IC = 4; N_OC = 4
    M0 = 0x4000; SH = 0; BIAS = 0   # out = round(acc / 4)

    rng = np.random.default_rng(42)
    W = rng.integers(-3, 4, (N_OC, N_IC, 3, 3), dtype=np.int8)
    # 3-row × 18-col input band: 16 output positions need cols 0..17 (kw=0..2 sliding)
    x = rng.integers(-8, 9, (N_IC, 3, 18), dtype=np.int8)

    # CPU golden: acc[oc,p] = sum_{ic,kh,kw} W[oc,ic,kh,kw] * x[ic,kh,p+kw]
    acc = np.zeros((N_OC, 16), dtype=np.int32)
    for oc in range(N_OC):
        for p in range(16):
            for ic in range(N_IC):
                for kh in range(3):
                    for kw in range(3):
                        acc[oc, p] += int(W[oc, ic, kh, kw]) * int(x[ic, kh, p + kw])
    gold = golden_rq(acc, M0, SH, BIAS)
    print(f'  acc [{acc.min()}, {acc.max()}]  gold [{gold.min()}, {gold.max()}]')

    # Pack ACT for R1: IN[ic*54 + kh*18 + col]
    act_buf = np.zeros(N_IC * 54, dtype=np.int8)
    for ic in range(N_IC):
        for kh in range(3):
            act_buf[ic*54 + kh*18 : ic*54 + kh*18 + 18] = x[ic, kh]

    # Pack WT: WT[(ic*9+tap)*17 + 1 + g] = W[g, ic, kh, kw]; tap = kh*3+kw
    WT_MAIN = N_IC * 9 * 17
    wt_buf = bytearray(WT_MAIN + N_OC * 7)
    for ic in range(N_IC):
        for kh in range(3):
            for kw in range(3):
                tap  = kh * 3 + kw
                base = (ic * 9 + tap) * 17
                wt_buf[base] = 0   # idx unused in R1 dense mode
                for g in range(N_OC):
                    wt_buf[base + 1 + g] = int(W[g, ic, kh, kw]) & 0xFF
    wt_buf[WT_MAIN:] = pack_rq(M0, SH, BIAS, N_OC)

    print(f'  ACT={len(act_buf)}B  WT={len(wt_buf)}B')
    dput(dma, ACT_OFF, bytes(act_buf))
    dput(dma, WT_OFF,  bytes(wt_buf))
    dput(dma, OUT_OFF, bytes(64))

    # Registers
    wr(regs, REG_CIN,         N_IC)
    wr(regs, REG_COUT,        N_OC)
    wr(regs, REG_HIN,         3)
    wr(regs, REG_WIN,         18)
    wr(regs, REG_N_CONFIG,    0)         # unused in R1 mode
    wr(regs, REG_POC,         0)
    wr(regs, REG_ACCU_MODE,   0)         # 1-in (SPEC 0 only)
    wr(regs, REG_APPLY_RELU,  0)
    wr(regs, REG_STRIDE,      1)
    wr(regs, REG_PAD,         1 << 10)  # bit10 = R1 fast path
    wr(regs, REG_LAYER_TYPE,  1)        # CONV_TYPE_3X3
    wr(regs, REG_ACT_CHUNK,   N_IC * 54)
    wr(regs, REG_WT_BLK_BYTES, len(wt_buf))
    wr(regs, REG_WT_BYTES,    len(wt_buf))
    wr(regs, REG_OUT_BYTES,   64)
    wr64(regs, REG_IN_ADDR_LO,  phys_base + ACT_OFF)
    wr64(regs, REG_WT_ADDR_LO,  phys_base + WT_OFF)
    wr64(regs, REG_OUT_ADDR_LO, phys_base + OUT_OFF)
    wr(regs, REG_N_OCB,       1)
    wr(regs, REG_N_POS,       1)
    wr(regs, REG_N_ICB,       1)
    wr(regs, REG_LAYER_MODE,  1)        # MUST be last; layer_seq does 1×1×1 = 1 kick

    wr(regs, REG_CTRL, AP_START)
    wait_done(regs)

    # Output layout: out_tile[oc][pos] sequential (flush_g outer, flush_p inner)
    out_raw = np.frombuffer(dget(dma, OUT_OFF, 64), dtype=np.int8).reshape(N_OC, 16)

    diff = np.abs(out_raw.astype(np.int16) - gold.astype(np.int16))
    print(f'  HW  [{out_raw.min():4d},{out_raw.max():4d}]: {out_raw.flatten()[:16].tolist()}')
    print(f'  CPU [{gold.min():4d},{gold.max():4d}]: {gold.flatten()[:16].tolist()}')
    print(f'  max_diff={diff.max()}  exact={100*(diff==0).mean():.0f}%  '
          f'within1={100*(diff<=1).mean():.0f}%')
    status = 'PASS' if diff.max() == 0 else ('CLOSE' if diff.max() <= 1 else 'FAIL')
    print(f'  → {status}')
    return status, diff.max()


# ── TEST 2: depthwise via swc FSM ────────────────────────────────────────────
def test_dw(regs, dma, phys_base):
    print('\n=== TEST 2: DW (4 ch, 16 positions, 3×18 band, n_cfg=1) ===')
    N_CH = 4
    M0 = 0x4000; SH = 0; BIAS = 0

    rng = np.random.default_rng(99)
    W = rng.integers(-3, 4, (N_CH, 3, 3), dtype=np.int8)   # (ch, kh, kw)
    x = rng.integers(-8, 9, (N_CH, 3, 18), dtype=np.int8)  # (ch, row, col)

    # CPU golden: acc[ch,p] = sum_{kh,kw} W[ch,kh,kw] * x[ch,kh,p+kw]
    acc = np.zeros((N_CH, 16), dtype=np.int32)
    for ch in range(N_CH):
        for p in range(16):
            for kh in range(3):
                for kw in range(3):
                    acc[ch, p] += int(W[ch, kh, kw]) * int(x[ch, kh, p + kw])
    gold = golden_rq(acc, M0, SH, BIAS)
    print(f'  acc [{acc.min()}, {acc.max()}]  gold [{gold.min()}, {gold.max()}]')

    # Pack ACT: IN[ch * tile_chw + kh * win + col],  tile_chw = 3*18 = 54
    # ch0: [0..53], ch1: [54..107], ch2: [108..161], ch3: [162..215]
    act_buf = np.zeros(N_CH * 54, dtype=np.int8)
    for ch in range(N_CH):
        for kh in range(3):
            act_buf[ch*54 + kh*18 : ch*54 + kh*18 + 18] = x[ch, kh]

    # Pack WT: tap * 17B per tap, then 4ch × 7B requant
    # entry[0]=idx=0, entry[1+g]=W[g,kh,kw] for g=0..3
    wt_buf = bytearray(9 * 17 + N_CH * 7)
    for kh in range(3):
        for kw in range(3):
            tap  = kh * 3 + kw
            base = tap * 17
            wt_buf[base] = 0   # idx unused
            for g in range(N_CH):
                wt_buf[base + 1 + g] = int(W[g, kh, kw]) & 0xFF
    wt_buf[9*17:] = pack_rq(M0, SH, BIAS, N_CH)

    print(f'  ACT={len(act_buf)}B  WT={len(wt_buf)}B')
    dput(dma, ACT_OFF, bytes(act_buf))
    dput(dma, WT_OFF,  bytes(wt_buf))
    dput(dma, OUT_OFF, bytes(64))

    tile_chw = 3 * 18   # reg_hin * reg_win = 54

    wr(regs, REG_CIN,         N_CH)
    wr(regs, REG_COUT,        N_CH)
    wr(regs, REG_HIN,         3)         # tile height (3 input rows)
    wr(regs, REG_WIN,         18)        # tile width  (18 input cols)
    wr(regs, REG_N_CONFIG,    1)         # n_cfg=1: dense (1 NZ per tap)
    wr(regs, REG_POC,         0)         # forced poc=0 by config_decoder for DW
    wr(regs, REG_ACCU_MODE,   0)         # forced 1-in for DW
    wr(regs, REG_APPLY_RELU,  0)
    wr(regs, REG_STRIDE,      1)
    wr(regs, REG_PAD,         0)         # no R1 mode
    wr(regs, REG_LAYER_TYPE,  2)         # CONV_TYPE_DW
    wr(regs, REG_ACT_CHUNK,   N_CH * tile_chw)
    wr(regs, REG_WT_BLK_BYTES, len(wt_buf))
    wr(regs, REG_WT_BYTES,    len(wt_buf))
    wr(regs, REG_OUT_BYTES,   64)
    wr64(regs, REG_IN_ADDR_LO,  phys_base + ACT_OFF)
    wr64(regs, REG_WT_ADDR_LO,  phys_base + WT_OFF)
    wr64(regs, REG_OUT_ADDR_LO, phys_base + OUT_OFF)
    wr(regs, REG_N_OCB,       1)
    wr(regs, REG_N_POS,       1)
    wr(regs, REG_N_ICB,       1)
    wr(regs, REG_LAYER_MODE,  1)        # MUST be last; layer_seq does 1×1×1 = 1 kick

    wr(regs, REG_CTRL, AP_START)
    wait_done(regs)

    out_raw = np.frombuffer(dget(dma, OUT_OFF, 64), dtype=np.int8).reshape(N_CH, 16)

    diff = np.abs(out_raw.astype(np.int16) - gold.astype(np.int16))
    print(f'  HW  [{out_raw.min():4d},{out_raw.max():4d}]: {out_raw.flatten()[:16].tolist()}')
    print(f'  CPU [{gold.min():4d},{gold.max():4d}]: {gold.flatten()[:16].tolist()}')
    print(f'  max_diff={diff.max()}  exact={100*(diff==0).mean():.0f}%  '
          f'within1={100*(diff<=1).mean():.0f}%')
    status = 'PASS' if diff.max() == 0 else ('CLOSE' if diff.max() <= 1 else 'FAIL')
    print(f'  → {status}')
    return status, diff.max()


# ── TEST 3: DW multi-kick state isolation ────────────────────────────────────
def test_dw_multikick(regs, dma, phys_base):
    """Two consecutive DW kicks with different input/output — verifies no state leak."""
    print('\n=== TEST 3: DW multi-kick state isolation ===')
    N_CH = 4
    M0 = 0x4000; SH = 0; BIAS = 0

    rng = np.random.default_rng(777)
    results = []
    for kick_i in range(2):
        W = rng.integers(-3, 4, (N_CH, 3, 3), dtype=np.int8)
        x = rng.integers(-8, 9, (N_CH, 3, 18), dtype=np.int8)
        acc = np.zeros((N_CH, 16), dtype=np.int32)
        for ch in range(N_CH):
            for p in range(16):
                for kh in range(3):
                    for kw in range(3):
                        acc[ch, p] += int(W[ch, kh, kw]) * int(x[ch, kh, p + kw])
        gold = golden_rq(acc, M0, SH, BIAS)

        act_buf = np.zeros(N_CH * 54, dtype=np.int8)
        for ch in range(N_CH):
            for kh in range(3):
                act_buf[ch*54 + kh*18 : ch*54 + kh*18 + 18] = x[ch, kh]
        wt_buf = bytearray(9 * 17 + N_CH * 7)
        for kh in range(3):
            for kw in range(3):
                tap = kh*3+kw; base = tap*17
                wt_buf[base] = 0
                for g in range(N_CH):
                    wt_buf[base+1+g] = int(W[g,kh,kw]) & 0xFF
        wt_buf[9*17:] = pack_rq(M0, SH, BIAS, N_CH)

        dput(dma, ACT_OFF, bytes(act_buf))
        dput(dma, WT_OFF,  bytes(wt_buf))
        dput(dma, OUT_OFF, bytes(64))

        wr(regs, REG_CIN,         N_CH)
        wr(regs, REG_COUT,        N_CH)
        wr(regs, REG_HIN,         3)
        wr(regs, REG_WIN,         18)
        wr(regs, REG_N_CONFIG,    1)
        wr(regs, REG_POC,         0)
        wr(regs, REG_ACCU_MODE,   0)
        wr(regs, REG_APPLY_RELU,  0)
        wr(regs, REG_STRIDE,      1)
        wr(regs, REG_PAD,         0)
        wr(regs, REG_LAYER_TYPE,  2)
        wr(regs, REG_ACT_CHUNK,   N_CH * 54)
        wr(regs, REG_WT_BLK_BYTES, len(wt_buf))
        wr(regs, REG_WT_BYTES,    len(wt_buf))
        wr(regs, REG_OUT_BYTES,   64)
        wr64(regs, REG_IN_ADDR_LO,  phys_base + ACT_OFF)
        wr64(regs, REG_WT_ADDR_LO,  phys_base + WT_OFF)
        wr64(regs, REG_OUT_ADDR_LO, phys_base + OUT_OFF)
        wr(regs, REG_N_OCB, 1); wr(regs, REG_N_POS, 1); wr(regs, REG_N_ICB, 1)
        wr(regs, REG_LAYER_MODE, 1)     # MUST be last
        wr(regs, REG_CTRL, AP_START)
        wait_done(regs)

        out_raw = np.frombuffer(dget(dma, OUT_OFF, 64), dtype=np.int8).reshape(N_CH, 16)
        diff = np.abs(out_raw.astype(np.int16) - gold.astype(np.int16))
        results.append(diff.max())
        print(f'  kick {kick_i}: max_diff={diff.max()}  exact={100*(diff==0).mean():.0f}%')

    status = 'PASS' if max(results) == 0 else 'FAIL'
    print(f'  → {status}')
    return status, max(results)


def dw_single_kick(regs, dma, phys_base, W, x, M0=0x4000, SH=0, BIAS=0, label=''):
    """Run one DW kick and return (out_raw, gold). No warmup needed if HW already init'd."""
    N_CH = 4
    acc = np.zeros((N_CH, 16), dtype=np.int32)
    for ch in range(N_CH):
        for p in range(16):
            for kh in range(3):
                for kw in range(3):
                    acc[ch, p] += int(W[ch, kh, kw]) * int(x[ch, kh, p + kw])
    gold = golden_rq(acc, M0, SH, BIAS)

    act_buf = np.zeros(N_CH * 54, dtype=np.int8)
    for ch in range(N_CH):
        for kh in range(3):
            act_buf[ch*54 + kh*18 : ch*54 + kh*18 + 18] = x[ch, kh]

    wt_buf = bytearray(9 * 17 + N_CH * 7)
    for kh in range(3):
        for kw in range(3):
            tap = kh*3+kw; base = tap*17
            wt_buf[base] = 0
            for g in range(N_CH):
                wt_buf[base+1+g] = int(W[g,kh,kw]) & 0xFF
    wt_buf[9*17:] = pack_rq(M0, SH, BIAS, N_CH)

    dput(dma, ACT_OFF, bytes(act_buf))
    dput(dma, WT_OFF,  bytes(wt_buf))
    dput(dma, OUT_OFF, bytes(64))

    wr(regs, REG_CIN,         N_CH); wr(regs, REG_COUT, N_CH)
    wr(regs, REG_HIN,         3);    wr(regs, REG_WIN,  18)
    wr(regs, REG_N_CONFIG,    1);    wr(regs, REG_POC,  0)
    wr(regs, REG_ACCU_MODE,   0);    wr(regs, REG_APPLY_RELU, 0)
    wr(regs, REG_STRIDE,      1);    wr(regs, REG_PAD,  0)
    wr(regs, REG_LAYER_TYPE,  2)
    wr(regs, REG_ACT_CHUNK,   N_CH * 54)
    wr(regs, REG_WT_BLK_BYTES, len(wt_buf))
    wr(regs, REG_WT_BYTES,    len(wt_buf))
    wr(regs, REG_OUT_BYTES,   64)
    wr64(regs, REG_IN_ADDR_LO,  phys_base + ACT_OFF)
    wr64(regs, REG_WT_ADDR_LO,  phys_base + WT_OFF)
    wr64(regs, REG_OUT_ADDR_LO, phys_base + OUT_OFF)
    wr(regs, REG_N_OCB, 1); wr(regs, REG_N_POS, 1); wr(regs, REG_N_ICB, 1)
    wr(regs, REG_LAYER_MODE, 1)
    wr(regs, REG_CTRL, AP_START)
    wait_done(regs)
    out_raw = np.frombuffer(dget(dma, OUT_OFF, 64), dtype=np.int8).reshape(N_CH, 16)
    diff = np.abs(out_raw.astype(np.int16) - gold.astype(np.int16))
    ok = 'PASS' if diff.max() == 0 else f'FAIL(max_diff={diff.max()})'
    print(f'  {label:30s} gold[{gold.min():4},{gold.max():4}] '
          f'hw[{out_raw.min():4},{out_raw.max():4}] {ok}')
    if diff.max() > 0:
        print(f'    gold: {gold.flatten()[:8].tolist()}')
        print(f'    hw:   {out_raw.flatten()[:8].tolist()}')
    return out_raw, gold, diff.max()


def diag_dw(regs, dma, phys_base):
    """Diagnostic suite for DW to pinpoint failure mode."""
    print('\n=== DIAG: DW path diagnostics ===')
    N_CH = 4

    # Test A: zero weights → expected all-zero output
    W0 = np.zeros((N_CH, 3, 3), dtype=np.int8)
    x0 = np.ones((N_CH, 3, 18), dtype=np.int8) * 5
    _, _, d = dw_single_kick(regs, dma, phys_base, W0, x0, label='A: zero weights, x=5')

    # Test B: identity - single center tap ch0=1, others 0; x=1 everywhere
    W1 = np.zeros((N_CH, 3, 3), dtype=np.int8); W1[0, 1, 1] = 4
    x1 = np.ones((N_CH, 3, 18), dtype=np.int8)
    # acc[0,all] = 4, acc[1,2,3,all] = 0; M0=0x4000 → out[0]=round(4/4)=1, out[1..3]=0
    _, _, d = dw_single_kick(regs, dma, phys_base, W1, x1, label='B: W[0,1,1]=4 x=1 → ch0=1')

    # Test C: all ch single tap (center) W=4, x=ones → all ch out=1
    W2 = np.zeros((N_CH, 3, 3), dtype=np.int8)
    for g in range(N_CH): W2[g, 1, 1] = 4
    _, _, d = dw_single_kick(regs, dma, phys_base, W2, x1, label='C: all ch W[g,1,1]=4 → out=1')

    # Test D: 9-tap sum W=1 everywhere, x=1 → acc=9 per ch → out=round(9/4)=2
    W3 = np.ones((N_CH, 3, 3), dtype=np.int8)
    _, _, d = dw_single_kick(regs, dma, phys_base, W3, x1, label='D: W=1 x=1 → acc=9 out=2')

    # Test E: use large M0 to see raw acc ≈ out (M0=65535,SH=0 → out≈acc/1)
    W4 = np.zeros((N_CH, 3, 3), dtype=np.int8); W4[0, 1, 1] = 1
    x4 = np.ones((N_CH, 3, 18), dtype=np.int8)
    # acc[0]=1, out≈round(1*65535/65536)=1 (identity scale)
    _, _, d = dw_single_kick(regs, dma, phys_base, W4, x4, M0=0xFFFF, SH=0,
                              label='E: W[0,1,1]=1 x=1 M0=0xFFFF → out≈1')

    # Test F: larger acc - W[0,1,1]=32, x=1; acc=32, with M0=0xFFFF out≈32
    W5 = np.zeros((N_CH, 3, 3), dtype=np.int8); W5[0, 1, 1] = 32
    _, _, d = dw_single_kick(regs, dma, phys_base, W5, x4, M0=0xFFFF, SH=0,
                              label='F: W[0,1,1]=32 x=1 M0=0xFFFF → out≈32')


if __name__ == '__main__':
    regs, dma, phys_base = open_hw()

    # Warmup with real conv1x1 config (LAYER_MODE=1 requires correct layout)
    print('[*] Warmup: running B0-style conv1x1 gate test...')
    # Minimal conv1x1: CIN=16, COUT=4, H=1, W=16, N_CFG=1, N_OCB=1, N_POS=1, N_ICB=1
    # ACT: 16ch × 16pos = 256B; WT: 1×17 + 4×7 = 45B (padded to 48B)
    wt_warm = bytearray(48)
    wt_warm[0]=0  # idx=0
    wt_warm[1]=1; wt_warm[2]=0; wt_warm[3]=0; wt_warm[4]=0  # s0g0..g3
    wt_warm[17] = 0xFF & 0x4000; wt_warm[18] = 0xFF & (0x4000 >> 8)  # ch0 M0=0x4000
    for i in range(4):  # all 4 OC requant at 17..17+28
        off = 17 + i*7
        wt_warm[off+0] = 0x00; wt_warm[off+1] = 0x40  # M0=0x4000
        wt_warm[off+2] = 0  # shift=0
    dput(dma, ACT_OFF, bytes(256)); dput(dma, WT_OFF, bytes(wt_warm)); dput(dma, OUT_OFF, bytes(64))
    wr(regs, REG_CIN, 16);   wr(regs, REG_COUT, 4)
    wr(regs, REG_HIN, 1);    wr(regs, REG_WIN, 16)
    wr(regs, REG_N_CONFIG, 1); wr(regs, REG_POC, 0)
    wr(regs, REG_ACCU_MODE, 2); wr(regs, REG_APPLY_RELU, 0)
    wr(regs, REG_PAD, 0);    wr(regs, REG_LAYER_TYPE, 0)
    wr(regs, REG_ACT_CHUNK, 256); wr(regs, REG_WT_BLK_BYTES, 48)
    wr(regs, REG_WT_BYTES, 48); wr(regs, REG_OUT_BYTES, 64)
    wr64(regs, REG_IN_ADDR_LO, phys_base + ACT_OFF)
    wr64(regs, REG_WT_ADDR_LO, phys_base + WT_OFF)
    wr64(regs, REG_OUT_ADDR_LO, phys_base + OUT_OFF)
    wr(regs, REG_N_OCB, 1); wr(regs, REG_N_POS, 1); wr(regs, REG_N_ICB, 1)
    wr(regs, REG_LAYER_MODE, 1)
    wr(regs, REG_CTRL, AP_START)
    wait_done(regs)
    print('[*] Warmup done')

    # Diagnostic first to understand failure
    diag_dw(regs, dma, phys_base)

    r1, d1 = test_c3x3_r1(regs, dma, phys_base)
    r2, d2 = test_dw(regs, dma, phys_base)
    r3, d3 = test_dw_multikick(regs, dma, phys_base)

    print(f'\n{"="*50}')
    print(f'conv3x3 R1 : {r1}  (max_diff={d1})')
    print(f'DW         : {r2}  (max_diff={d2})')
    print(f'DW multi   : {r3}  (max_diff={d3})')
    all_pass = all(s == 'PASS' for s in [r1, r2, r3])
    print('ALL PASS' if all_pass else 'SOME FAIL')
    sys.exit(0 if all_pass else 1)
