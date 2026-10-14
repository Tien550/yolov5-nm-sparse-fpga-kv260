#!/usr/bin/env python3
"""
fpga_conv3x3.py — conv3x3 on the unified IP (Rev43: multi-input-channel), full-layer
host orchestration.

One kick computes: 4 output channels (= the 4 GROUPS of SPEC0, out_ch=flush_g 0..3)
× 16 spatial positions (one output row, 16 cols), accumulating over ALL Cin input
channels (the in-RTL ic loop) and the 9 kernel taps.

Layouts the RTL expects:
  IN BRAM  (CHW): channel c at c*tile_chw (tile_chw = Hin*Win). For one output row
                  with Wout=16: Hin=3, Win=18 → tile_chw=54.  addr = c*54 + kh*18 + kw + p
  WT BRAM       : entry (in_ch c, tap t) at (c*9 + t)*n_cfg*17, 17 bytes:
                  [idx=0, s0g0,s0g1,s0g2,s0g3, s1..s3=0]. s0g{g} = W[out_ch g][c][t].
                  requant tail (7B/ch × 4) at offset Cin*9*n_cfg*17.

Requant lag (Rev42): a kick applies the PREVIOUS kick's requant. Handled by 1 prime
kick per out-ch-block (same requant across the block's spatial tiles).

Full layer: loop out-ch-blocks (ceil(Cout/4)) × output rows × col-blocks(ceil(Wout/16)).
"""
import ctypes, mmap, os, struct, time
# Under sudo, root misses the user's fast numpy 1.26.4 (~/.local) and falls back to system
# numpy 1.21.5 with reference (non-NEON) BLAS — ~40x slower GEMM. Prepend the user site before
# importing numpy so self-tests/inference under sudo use the fast OpenBLAS. See project_sudo_numpy_blas.
import sys as _sys
if os.geteuid() == 0:
    _us = f"/home/ubuntu/.local/lib/python{_sys.version_info.major}.{_sys.version_info.minor}/site-packages"
    if os.path.isdir(_us) and _us not in _sys.path: _sys.path.insert(0, _us)
import numpy as np

REG_AP_CTRL=0x000; REG_CIN=0x040; REG_COUT=0x050; REG_HIN=0x060; REG_WIN=0x070
REG_N_CONFIG=0x080; REG_POC=0x090; REG_ACCU_MODE=0x0A0; REG_APPLY_RELU=0x0B0
REG_WT_LO=0x0C0; REG_IN_LO=0x0E0; REG_OUT_LO=0x100; REG_STRIDE=0x120; REG_PAD=0x130
REG_LAYER_TYPE=0x140; REG_OUT_BYTES=0x150; REG_WT_BYTES=0x160; REG_DBG_STATE=0x170
# B0 layer_sequencer registers (on-chip layer loop)
REG_LAYER_MODE=0x190; REG_N_OCB=0x1A0; REG_N_POS=0x1B0; REG_N_ICB=0x1C0; REG_WT_BLK_BYTES=0x1D0
REG_ACT_CHUNK=0x1E0   # per-tile ACT bytes/stride (1024=conv3x3, 256=conv1x1 R3)
# Network sequencer (LT14, Rev132 DESCRIPTOR mode): reuses REG_IN_LO=desc_base, REG_CIN=n_layers.
# B0 DDR tile-stream regions (distinct; WT@16MB has 48MB for 512→512: N_OCB*N_ICB*4096=16MB)
B0_ACT=0x0000000; B0_OUT=0x0800000; B0_WT=0x1000000   # ACT@0(8MB) OUT@8MB(8MB) WT@16MB(48MB)
AP_START=0x1; AP_DONE=0x2; AP_IDLE=0x4; AXI_SIZE=0x1000; DMA_TOTAL=64*1024*1024
OFF_ACT=0x000000; OFF_WT=0x100000; OFF_OUT=0x200000
KW=18  # input cols per kick (16 outputs + 2 halo)
KH=3   # input rows per kick (1 output row)
WOUT=16

class HW:
    def __init__(self):
        fd=os.open("/dev/uio4",os.O_RDWR|os.O_SYNC)
        m=mmap.mmap(fd,AXI_SIZE,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,offset=0)
        self._r=(ctypes.c_uint32*(AXI_SIZE//4)).from_buffer(m); os.close(fd)
        with open("/sys/module/dma_region/parameters/phys_base") as f: self.phys=int(f.read().strip())
        dfd=os.open("/dev/dma_region",os.O_RDWR|os.O_SYNC)
        self.dma=mmap.mmap(dfd,DMA_TOTAL,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE); os.close(dfd)
        # bind the DDR buffer to a ctypes base address ONCE (avoid re-creating a
        # from_buffer view every put — that ctypes type creation, not the byte copy,
        # is ~50us of the old per-put 60us). memmove dst stays base+off = identical
        # bytes & coherency as the old per-call view → safe (does NOT touch kick/get).
        self._dma_base = ctypes.addressof((ctypes.c_char*DMA_TOTAL).from_buffer(self.dma))
    def wr(self,o,v): self._r[o>>2]=int(v)&0xFFFFFFFF
    def rd(self,o): return self._r[o>>2]
    def wr64(self,o,a):
        a=int(a); self._r[o>>2]=a&0xFFFFFFFF; self._r[(o+0x10)>>2]=(a>>32)&0xFFFFFFFF
    def put(self,off,arr):
        d=np.ascontiguousarray(arr,dtype=np.int8)
        ctypes.memmove(self._dma_base+off, d.ctypes.data, d.nbytes)  # dst==base+off (same as old view)
    def dst_ptr(self,off,maxn=1<<20):
        """Cached DDR destination pointer for fast per-kick memmove (avoids re-creating
        the ctypes view + np.ascontiguousarray every put — see throughput profile)."""
        if not hasattr(self,'_dstc'): self._dstc={}
        if off not in self._dstc:
            self._dstc[off]=ctypes.addressof((ctypes.c_char*maxn).from_buffer(self.dma,off))
        return self._dstc[off]
    def put_fast(self,dst,arr):  # arr must be contiguous int8
        ctypes.memmove(dst, arr.ctypes.data, arr.nbytes)
    def get(self,off,n):
        # NOTE: per-layer reads are small/frequent → ThreadPoolExecutor dispatch overhead exceeds the
        # WC-read latency gain (measured deint 73→103ms with threading). Single-thread bytes() is best.
        return np.frombuffer(bytes(self.dma[off:off+n]),dtype=np.int8)
    def kick(self,to=5.0):
        # TIGHT-POLL (no sleep): ap_idle is usually asserted immediately after the previous kick's
        # dbg_state clears, so sleep(1ms) overshot badly per kick. Tight poll with rare yield (every
        # 4096 iters) keeps timeout safety without the 1ms granularity floor. Direct reg index = fast.
        r=self._r; ap=REG_AP_CTRL>>2; dbg=REG_DBG_STATE>>2
        i=0
        while not (r[ap]&AP_IDLE):
            i+=1
            if i&0xFFF==0: time.sleep(0.0002)
            if i>500000: raise TimeoutError("no ap_idle")
        r[ap]=AP_START
        i=0
        while not (r[ap]&AP_DONE):
            i+=1
            if i>4000000: raise TimeoutError("ap_done timeout")
        i=0
        while r[dbg]&0xF:
            i+=1
            if i>2000000: break
    def kick_fast(self):
        """Minimal-overhead kick: tight poll loops, timeout checked every 4096 iters."""
        r=self._r; ap=REG_AP_CTRL>>2; dbg=REG_DBG_STATE>>2
        i=0
        while not (r[ap]&AP_IDLE):
            i+=1
            if i&0xFFF==0: time.sleep(0.0005)
            if i>200000: raise TimeoutError("no ap_idle")
        r[ap]=AP_START
        i=0
        while not (r[ap]&AP_DONE):
            i+=1
            if i>2000000: raise TimeoutError("ap_done timeout")
        i=0
        while r[dbg]&0xF:
            i+=1
            if i>2000000: break

def encode_block(w, m0, shift, bias, n_cfg=1):
    """w[oc(<=4), ic, 3, 3] int8; m0/shift/bias[oc]. → blob (Cin*9*17 + 4*7 requant).
    Vectorized (no per-byte Python loop): entry (c,tap) at (c*9+tap)*17, byte0=idx=0,
    bytes1..1+oc_n = s0g{g}=w[g,c,kh,kw], rest 0."""
    oc_n, Cin = w.shape[0], w.shape[1]
    ent = np.zeros((Cin*9, 17), np.uint8)
    # w[oc_n,Cin,3,3] → [Cin,9,oc_n] (tap=kh*3+kw) → [Cin*9, oc_n]
    wr = w.reshape(oc_n, Cin, 9).transpose(1, 2, 0).reshape(Cin*9, oc_n)
    ent[:, 1:1+oc_n] = wr.astype(np.int8).view(np.uint8)
    tail = np.zeros((4, 7), np.uint8)
    for g in range(4):
        mm = int(m0[g]) if g < oc_n else 0x4000
        ss = int(shift[g]) if g < oc_n else 15
        bb = int(bias[g]) & 0xFFFFFFFF if g < oc_n else 0
        tail[g] = [mm&0xFF, (mm>>8)&0xFF, ss&0xFF, bb&0xFF, (bb>>8)&0xFF, (bb>>16)&0xFF, (bb>>24)&0xFF]
    return ent.tobytes() + tail.tobytes()

CIN_CHUNK = 16   # max input channels per kick (fits IN BRAM 1024 / tile_chw=54, WT BRAM)
R1_MODE = False  # Rev47: set True → PAD bit10 routes conv3x3 kicks to conv_kernel_r1 (R1
                 # pipelined feeder, ~3.7× compute). Only conv3x3/conv1x1 (via _pad_ctrl);
                 # dwconv_layer writes PAD=1 directly so stays on the old path.
DBG_WT = False   # Rev48 DEBUG: PAD bit11 → conv_kernel_r1 stores LOADED center weights to
                 # OUT BRAM instead of conv result (read back wt_tile to find stale chunks).
def _pad_ctrl(keep, noflush): return 1 | (int(keep)<<8) | (int(noflush)<<9) | (int(R1_MODE)<<10) | (int(DBG_WT)<<11)

def conv3x3_tile(hw, act_tile, w_block, m0, shift, bias, n_cfg=1, relu=1, prime=True):
    """act_tile[Cin,KH,KW], w_block[oc_n,Cin,3,3], m0/shift/bias[oc_n] → out[oc_n,WOUT].
    Cin>CIN_CHUNK → channel-tiling: split into chunks, accumulate in RTL psum across kicks
    (keep_psum/no_flush via PAD[8]/[9]); only the last chunk flushes. prime handles requant lag."""
    Cin = act_tile.shape[0]; oc_n = w_block.shape[0]
    chunks = [(i, min(i+CIN_CHUNK, Cin)) for i in range(0, Cin, CIN_CHUNK)]; nch=len(chunks)
    def kick_chunk(c0, c1, keep, noflush):
        cin_c = c1-c0
        blob = encode_block(w_block[:, c0:c1], m0, shift, bias, n_cfg)
        a = np.ascontiguousarray(act_tile[c0:c1].reshape(cin_c, KH*KW).reshape(-1), np.int8)
        hw.put(OFF_ACT, a); hw.put(OFF_WT, np.frombuffer(blob,np.int8))
        if not keep: hw.put(OFF_OUT, np.full(64,0,np.int8))
        hw.wr(REG_LAYER_TYPE,1); hw.wr(REG_CIN,cin_c); hw.wr(REG_COUT,4); hw.wr(REG_HIN,KH); hw.wr(REG_WIN,KW)
        hw.wr(REG_N_CONFIG,n_cfg); hw.wr(REG_POC,2); hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu)
        hw.wr(REG_STRIDE,1); hw.wr(REG_PAD,_pad_ctrl(keep,noflush))
        hw.wr64(REG_IN_LO,hw.phys+OFF_ACT); hw.wr64(REG_WT_LO,hw.phys+OFF_WT); hw.wr64(REG_OUT_LO,hw.phys+OFF_OUT)
        hw.wr(REG_OUT_BYTES,64); hw.wr(REG_WT_BYTES,len(blob)); hw.kick()
    if prime: kick_chunk(chunks[0][0], chunks[0][1], False, False)   # prime: load requant
    for idx,(c0,c1) in enumerate(chunks):
        kick_chunk(c0, c1, keep=(idx>0), noflush=(idx < nch-1))
    return hw.get(OFF_OUT,64).reshape(4,16)[:oc_n, :WOUT]

WT_STRIDE = 0x1000   # per-chunk WT DDR offset (≤4096B/chunk: 16*9*17+28=2476)
def conv3x3_layer(hw, act, w, m0, shift, bias, relu=1, pad=1, stride=1):
    """act[Cin,H,W], w[Cout,Cin,3,3] → out[Cout,Hout,Wout]. Optimized: pre-encode
    weight blobs per out-ch-block at distinct WT DDR offsets (put once, per-kick just
    switches WT_LO — no re-encode/re-DMA of weights), const regs once/block.

    stride==2: compute ONLY the output rows we keep (padded input row 2*oi) instead of
    every row then [::2,::2] — halves the per-layer kicks for stride-2 convs (#1). The
    IP emits stride-1 cols per kick, so columns are computed full-width then subsampled
    (#2 = native col stride needs RTL). Bit-identical to stride-1-then-[::2,::2] because
    each kept kick computes the exact same window — verified: ref stride2 == ref stride1[::2,::2]."""
    Cout, Cin, _, _ = w.shape; _, H, W = act.shape
    ap = np.pad(act, ((0,0),(pad,pad),(pad,pad))).astype(np.int8)
    Hout = (H + 2*pad - 3)//stride + 1
    Wout = (W + 2*pad - 3)//stride + 1
    out = np.zeros((Cout, Hout, Wout), np.int8)
    chunks = [(i, min(i+CIN_CHUNK, Cin)) for i in range(0, Cin, CIN_CHUNK)]; nch = len(chunks)
    n_cb = (W + WOUT - 1)//WOUT          # col blocks over full (stride-1) width
    for oc0 in range(0, Cout, 4):
        oc1 = min(oc0+4, Cout); oc_n = oc1-oc0
        # pre-encode each chunk's weights ONCE to a distinct WT DDR offset (constant
        # across all spatial tiles; per-kick only switches WT_LO — no re-encode/re-DMA)
        blens=[]
        for idx,(c0,c1) in enumerate(chunks):
            blob = encode_block(w[oc0:oc1,c0:c1], m0[oc0:oc1], shift[oc0:oc1], bias[oc0:oc1])
            hw.put(OFF_WT + idx*WT_STRIDE, np.frombuffer(blob,np.int8)); blens.append(len(blob))
        # constant registers (set once per out-ch-block)
        hw.wr(REG_LAYER_TYPE,1); hw.wr(REG_COUT,4); hw.wr(REG_HIN,KH); hw.wr(REG_WIN,KW)
        hw.wr(REG_N_CONFIG,1); hw.wr(REG_POC,2); hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu); hw.wr(REG_STRIDE,1)
        hw.wr64(REG_IN_LO,hw.phys+OFF_ACT); hw.wr64(REG_OUT_LO,hw.phys+OFF_OUT); hw.wr(REG_OUT_BYTES,64)
        wtlo = [hw.phys+OFF_WT+idx*WT_STRIDE for idx in range(nch)]
        pads = [_pad_ctrl(idx>0, idx<nch-1) for idx in range(nch)]
        def kick_chunk(idx, c0, c1, aflat):
            hw.put(OFF_ACT, aflat); hw.wr64(REG_WT_LO, wtlo[idx])  # hw.put: reliable (memmove
            hw.wr(REG_CIN,c1-c0); hw.wr(REG_WT_BYTES,blens[idx]); hw.wr(REG_PAD,pads[idx])  # commits
            hw.kick()                                          # to DDR before ap_start)
        # prime once per block (requant lag): chunk0 of first tile, keep=0 noflush=0
        c0,c1=chunks[0]; t0=np.zeros((c1-c0,KH,KW),np.int8); s=ap[c0:c1,0:KH,0:KW]; t0[:,:s.shape[1],:s.shape[2]]=s
        hw.put(OFF_ACT, t0.reshape(-1)); hw.wr64(REG_WT_LO,wtlo[0]); hw.wr(REG_CIN,c1-c0)
        hw.wr(REG_WT_BYTES,blens[0]); hw.wr(REG_PAD,_pad_ctrl(False,False)); hw.kick()
        for oi in range(Hout):
            ri = oi*stride                       # padded input row start (stride1: ri==oi)
            rowbuf = np.zeros((oc_n, W), np.int8) # stride-1 cols for this output row
            for cb in range(n_cb):
                x0=cb*WOUT; npos=min(WOUT,W-x0)
                tf=np.zeros((Cin,KH,KW),np.int8); s=ap[:,ri:ri+KH,x0:x0+KW]; tf[:,:s.shape[1],:s.shape[2]]=s
                for idx,(c0,c1) in enumerate(chunks):
                    kick_chunk(idx, c0, c1, tf[c0:c1].reshape(-1))
                rowbuf[:, x0:x0+npos] = hw.get(OFF_OUT,64).reshape(4,16)[:oc_n,:npos]
            out[oc0:oc1, oi, :] = rowbuf[:, ::stride][:, :Wout] if stride>1 else rowbuf
    return out

# Cache for encoded WT blobs, keyed by weight-array identity (id) + shape. Weights are
# constant across inference frames (video stream) → encode once per layer, reuse every
# frame. Removes the dominant host overhead (~200ms/deep-conv1x1 of encode_block loops
# found by prof_host.py). put() to DDR still happens each frame (cheap, ~1ms).
_WT_CACHE = {}

def conv_layer_B0(hw, act, w, m0, shift, bias, relu=1, pad=1, stride=1, to=30.0):
    """B0 (Rev51): whole conv3x3 layer in ONE ap_start via the on-chip layer_sequencer.
    Host prepares (1x/layer): ACT tile-stream + per-(oc,ic) WT blobs (chunks padded to 16
    channels → uniform kick), fires 1 kick, reads OUT tile-stream, de-interleaves. The HW
    loops every (oc-block, position, ic-chunk) — no host per-kick loop. Bit-exact to the
    per-tile conv3x3_layer (same tiles/weights). Requires LAYER_MODE register + R1 kernel."""
    Cout, Cin, _, _ = w.shape; _, H, W = act.shape
    ap = np.pad(act, ((0,0),(pad,pad),(pad,pad))).astype(np.int8)
    Hout = (H + 2*pad - 3)//stride + 1
    Wout = (W + 2*pad - 3)//stride + 1
    chunks = [(i, min(i+CIN_CHUNK, Cin)) for i in range(0, Cin, CIN_CHUNK)]
    N_ICB = len(chunks); n_cb = (W + WOUT - 1)//WOUT
    N_POS = Hout * n_cb; N_OCB = (Cout + 3)//4
    # 4KB-safe strides (dma_engine doesn't clamp bursts to 4KB): ACT 1024 (tile 864), WT 4096
    # (blob 2476). Both divide 4096 so 256B bursts never cross a 4KB boundary.
    TILE_BYTES = CIN_CHUNK*KH*KW       # 864 real ACT bytes/tile
    ACT_STRIDE = 1024; WBB = 4096      # must match REG_ACT_CHUNK + REG_WT_BLK_BYTES

    # ── ACT tile-stream [pos][ic][ACT_STRIDE] (vectorized sliding-window im2col) ──
    # Pad ap so every (oi*stride, cb*WOUT) window of (KH,KW) is in-bounds + channels→N_ICB*16,
    # then sliding_window_view + subsample(stride rows, WOUT cols) + reshape. Bit-equivalent to
    # the per-(oi,cb,ic) loop; out-of-bounds = zero-pad (same as the old tile pre-zero).
    from numpy.lib.stride_tricks import sliding_window_view
    Cp = N_ICB*CIN_CHUNK
    Hneed = max((Hout-1)*stride + KH, ap.shape[1]); Wneed = max((n_cb-1)*WOUT + KW, ap.shape[2])
    app = np.zeros((Cp, Hneed, Wneed), np.int8); app[:Cin, :ap.shape[1], :ap.shape[2]] = ap
    swv = sliding_window_view(app, (KH, KW), axis=(1, 2))      # [Cp, H', W', KH, KW]
    sel = swv[:, ::stride, ::WOUT][:, :Hout, :n_cb]            # [Cp, Hout, n_cb, KH, KW]
    act_t = np.ascontiguousarray(
        sel.reshape(N_ICB, CIN_CHUNK, Hout, n_cb, KH, KW).transpose(2, 3, 0, 1, 4, 5)
    ).reshape(N_POS, N_ICB, TILE_BYTES)
    act_s = np.zeros((N_POS, N_ICB, ACT_STRIDE), np.int8)
    act_s[:, :, :TILE_BYTES] = act_t
    hw.put(B0_ACT, act_s.reshape(-1))

    # ── WT blobs [oc-block][ic-chunk], each encode_block of 16-padded channels, padded → WBB ──
    # Cached by weight identity: encode once, reuse every frame (see _WT_CACHE).
    _wk = (id(w), 'c3x3b0', Cout, Cin)
    _wb = _WT_CACHE.get(_wk)
    if _wb is None:
        wt_s = np.zeros((N_OCB, N_ICB, WBB), np.uint8)
        for ocb in range(N_OCB):
            oc0 = ocb*4; oc1 = min(oc0+4, Cout)
            for ic,(c0,c1) in enumerate(chunks):
                wp = np.zeros((oc1-oc0, CIN_CHUNK, 3, 3), np.int8)  # pad channels → 16
                wp[:, :c1-c0] = w[oc0:oc1, c0:c1]
                blob = encode_block(wp, m0[oc0:oc1], shift[oc0:oc1], bias[oc0:oc1])
                wt_s[ocb, ic, :len(blob)] = np.frombuffer(blob, np.uint8)
        _wb = wt_s.reshape(-1).view(np.int8)
        _WT_CACHE[_wk] = _wb
    hw.put(B0_WT, _wb)

    # ── registers (kernel cfg + B0 loop cfg) ──
    hw.wr(REG_LAYER_TYPE,1); hw.wr(REG_CIN,CIN_CHUNK); hw.wr(REG_COUT,4)
    hw.wr(REG_HIN,KH); hw.wr(REG_WIN,KW); hw.wr(REG_N_CONFIG,1); hw.wr(REG_POC,2)
    hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu); hw.wr(REG_STRIDE,1)
    hw.wr(REG_PAD, 1<<10)                                  # r1_mode (B0 uses R1 kernel)
    hw.wr64(REG_IN_LO, hw.phys+B0_ACT); hw.wr64(REG_WT_LO, hw.phys+B0_WT)
    hw.wr64(REG_OUT_LO, hw.phys+B0_OUT); hw.wr(REG_OUT_BYTES,64); hw.wr(REG_WT_BYTES,WBB)
    hw.wr(REG_N_OCB,N_OCB); hw.wr(REG_N_POS,N_POS); hw.wr(REG_N_ICB,N_ICB); hw.wr(REG_WT_BLK_BYTES,WBB)
    hw.wr(REG_ACT_CHUNK,ACT_STRIDE)
    hw.wr(REG_LAYER_MODE,1)
    hw.kick(to=to)                                        # ONE kick = whole layer
    hw.wr(REG_LAYER_MODE,0); hw.wr(REG_ACT_CHUNK,1024)    # restore defaults

    # ── read OUT tile-stream [oc-block][pos][4,16] → de-interleave (vectorized) ──
    out_s = hw.get(B0_OUT, N_OCB*N_POS*64).reshape(N_OCB, N_POS, 4, 16)
    out_s1 = np.ascontiguousarray(                          # stride-1 cols, then subsample
        out_s.reshape(N_OCB, Hout, n_cb, 4, WOUT).transpose(0, 3, 1, 2, 4)
             .reshape(N_OCB*4, Hout, n_cb*WOUT)[:Cout, :, :W])
    return out_s1[:, :, ::stride][:, :, :Wout] if stride > 1 else out_s1

def encode_block_1x1(w, m0, shift, bias):
    """R3: 1x1 weight blob. w[oc_n(<=4), Cin] int8 → (Cin*17 + 4*7) bytes. Entry (ch) at ch*17:
    byte0=idx=0, bytes1..1+oc_n = s0g{g}=w[g,ch] (1 tap); requant tail at Cin*17 (7B/g)."""
    oc_n, Cin = w.shape
    ent = np.zeros((Cin, 17), np.uint8)
    ent[:, 1:1+oc_n] = w.T.astype(np.int8).view(np.uint8)        # [oc_n,Cin] → [Cin,oc_n]
    tail = np.zeros((4, 7), np.uint8)
    for g in range(4):
        mm = int(m0[g]) if g < oc_n else 0x4000
        ss = int(shift[g]) if g < oc_n else 15
        bb = int(bias[g]) & 0xFFFFFFFF if g < oc_n else 0
        tail[g] = [mm&0xFF,(mm>>8)&0xFF,ss&0xFF,bb&0xFF,(bb>>8)&0xFF,(bb>>16)&0xFF,(bb>>24)&0xFF]
    return ent.tobytes() + tail.tobytes()

def conv1x1_layer_B0(hw, act, w1x1, m0, shift, bias, relu=1, to=30.0):
    """R3 (Rev54): whole conv1x1 layer in ONE ap_start via layer_sequencer + 1-tap R1 kernel.
    Smaller tiles than conv_layer_B0: ACT 16ch×16col=256B (no halo), WT blob Cin*17+28 (1 tap,
    pad→512). ~6.5× less DMA/kick than the 3x3-center-tap path. w1x1[Cout,Cin,1,1] or [Cout,Cin]."""
    _worig_id = id(w1x1)                    # WT cache key: original weight identity (pre-reshape)
    w1x1 = w1x1.reshape(w1x1.shape[0], w1x1.shape[1])
    Cout, Cin = w1x1.shape; _, H, W = act.shape
    act = act.astype(np.int8)
    CC = 32                                  # conv1x1 CIN_CHUNK=32 (kernel widened Rev65; ÷2 N_ICB).
    chunks = [(i, min(i+CC, Cin)) for i in range(0, Cin, CC)]
    N_ICB = len(chunks); n_cb = (W + WOUT - 1)//WOUT
    N_POS = H * n_cb; N_OCB = (Cout + 3)//4
    ACT_STRIDE = CC*WOUT; WBB = 1024         # 32×16=512 ACT, blob 32×17+28=572→1024 (both | 4096)

    # ── ACT tile-stream [pos][ic][16ch,16col] (no halo; channels padded to 16) ──
    # Vectorized (no Python loop): pad channels→N_ICB*16 and cols→n_cb*16, then
    # reshape/transpose to [pos=row*n_cb+cb, ic, ch*16+col]. Bit-equivalent to the
    # per-(row,cb,ic) loop, ~10× faster (no per-tile alloc).
    act_p = np.zeros((N_ICB*CC, H, n_cb*WOUT), np.int8)
    act_p[:Cin, :, :W] = act
    act_s = np.ascontiguousarray(
        act_p.reshape(N_ICB, CC, H, n_cb, WOUT).transpose(2, 3, 0, 1, 4)
    ).reshape(N_POS, N_ICB, ACT_STRIDE)
    hw.put(B0_ACT, act_s.reshape(-1))

    # ── WT blobs [oc-block][ic-chunk] = encode_block_1x1 (16-padded channels) ──
    # Cached by weight identity: encode once, reuse every frame (see _WT_CACHE).
    _wk = (_worig_id, 'c1x1b0', Cout, Cin)
    _wb = _WT_CACHE.get(_wk)
    if _wb is None:
        wt_s = np.zeros((N_OCB, N_ICB, WBB), np.uint8)
        for ocb in range(N_OCB):
            oc0 = ocb*4; oc1 = min(oc0+4, Cout)
            for ic,(c0,c1) in enumerate(chunks):
                wp = np.zeros((oc1-oc0, CC), np.int8)
                wp[:, :c1-c0] = w1x1[oc0:oc1, c0:c1]
                blob = encode_block_1x1(wp, m0[oc0:oc1], shift[oc0:oc1], bias[oc0:oc1])
                wt_s[ocb, ic, :len(blob)] = np.frombuffer(blob, np.uint8)
        _wb = wt_s.reshape(-1).view(np.int8)
        _WT_CACHE[_wk] = _wb
    hw.put(B0_WT, _wb)

    # ── registers ──
    hw.wr(REG_LAYER_TYPE,0); hw.wr(REG_CIN,CC); hw.wr(REG_COUT,4)
    hw.wr(REG_HIN,1); hw.wr(REG_WIN,WOUT); hw.wr(REG_N_CONFIG,1); hw.wr(REG_POC,2)
    hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu); hw.wr(REG_STRIDE,1)
    hw.wr(REG_PAD, 1<<10)                                  # r1_mode (mode_1x1 derived from LAYER_TYPE=0)
    hw.wr64(REG_IN_LO, hw.phys+B0_ACT); hw.wr64(REG_WT_LO, hw.phys+B0_WT)
    hw.wr64(REG_OUT_LO, hw.phys+B0_OUT); hw.wr(REG_OUT_BYTES,64); hw.wr(REG_WT_BYTES,WBB)
    hw.wr(REG_N_OCB,N_OCB); hw.wr(REG_N_POS,N_POS); hw.wr(REG_N_ICB,N_ICB)
    hw.wr(REG_WT_BLK_BYTES,WBB); hw.wr(REG_ACT_CHUNK,ACT_STRIDE)
    hw.wr(REG_LAYER_MODE,1)
    hw.kick(to=to)
    hw.wr(REG_LAYER_MODE,0); hw.wr(REG_ACT_CHUNK,1024)     # restore defaults

    # ── read OUT tile-stream → [Cout,H,W] (vectorized de-interleave) ──
    out_s = hw.get(B0_OUT, N_OCB*N_POS*64).reshape(N_OCB, N_POS, 4, 16)
    out = np.ascontiguousarray(
        out_s.reshape(N_OCB, H, n_cb, 4, WOUT).transpose(0, 3, 1, 2, 4)
             .reshape(N_OCB*4, H, n_cb*WOUT)[:Cout, :, :W])
    return out

def dwconv_tile(hw, act_block, w_block, m0, shift, bias, relu=1, prime=True):
    """Depthwise 4 channels/kick (Rev45): act_block[4,KH,KW], w_block[4,3,3],
    m0/shift/bias[4] → out[4,WOUT]. The 4 GROUPS = 4 channels (no cross-channel sum)."""
    nch = act_block.shape[0]
    a = np.zeros((4, KH, KW), np.int8); a[:nch] = act_block      # CHW: channel g at g*tile_chw
    a = np.ascontiguousarray(a.reshape(-1), np.int8)
    blob = bytearray()
    for t in range(9):
        kh,kw=t//3,t%3
        e=bytearray(17); e[0]=0
        for g in range(4):
            e[1+g] = struct.pack('b', int(w_block[g,kh,kw]))[0] if g<nch else 0
        blob += e
    for g in range(4):                                          # requant tail (4 channels)
        if g<nch: blob += struct.pack('<H',int(m0[g]))+struct.pack('B',int(shift[g]))+struct.pack('<i',int(bias[g]))
        else:     blob += struct.pack('<H',0x4000)+struct.pack('B',15)+struct.pack('<i',0)
    def kick():
        hw.put(OFF_ACT,a); hw.put(OFF_WT,np.frombuffer(bytes(blob),np.int8)); hw.put(OFF_OUT,np.full(64,0,np.int8))
        hw.wr(REG_LAYER_TYPE,2); hw.wr(REG_CIN,4); hw.wr(REG_COUT,4); hw.wr(REG_HIN,KH); hw.wr(REG_WIN,KW)
        hw.wr(REG_N_CONFIG,1); hw.wr(REG_POC,0); hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu)
        hw.wr(REG_STRIDE,1); hw.wr(REG_PAD,1)
        hw.wr64(REG_IN_LO,hw.phys+OFF_ACT); hw.wr64(REG_WT_LO,hw.phys+OFF_WT); hw.wr64(REG_OUT_LO,hw.phys+OFF_OUT)
        hw.wr(REG_OUT_BYTES,64); hw.wr(REG_WT_BYTES,len(blob)); hw.kick()
    if prime: kick()
    kick()
    return hw.get(OFF_OUT,64).reshape(4,16)[:nch, :WOUT]

def _dw_encode(wb, m0b, shb, bb, nch):
    """dwconv blob: 9 taps (s0g{g}=W[g][tap]) + 4-ch requant tail. Vectorized."""
    ent = np.zeros((9,17), np.uint8)
    ent[:, 1:1+nch] = wb.reshape(nch,9).T.astype(np.int8).view(np.uint8)   # [9,nch]
    tail = np.zeros((4,7), np.uint8)
    for g in range(4):
        mm=int(m0b[g]) if g<nch else 0x4000; ss=int(shb[g]) if g<nch else 15; bb_=int(bb[g])&0xFFFFFFFF if g<nch else 0
        tail[g]=[mm&0xFF,(mm>>8)&0xFF,ss&0xFF,bb_&0xFF,(bb_>>8)&0xFF,(bb_>>16)&0xFF,(bb_>>24)&0xFF]
    return np.frombuffer(ent.tobytes()+tail.tobytes(), np.int8)

def dwconv_layer(hw, act, w, m0, shift, bias, relu=1, pad=1):
    """Depthwise layer: act[C,H,W], w[C,3,3] → out[C,H,W]. 4 channels/kick (optimized:
    pre-encode dw blob/block, const regs once, kick_fast, prime once)."""
    C, H, W = act.shape
    ap = np.pad(act, ((0,0),(pad,pad),(pad,pad))).astype(np.int8)
    out = np.zeros((C, H, W), np.int8); n_cb = (W + WOUT - 1)//WOUT
    for c0 in range(0, C, 4):
        c1 = min(c0+4, C); nch = c1-c0
        blob = _dw_encode(w[c0:c1], m0[c0:c1], shift[c0:c1], bias[c0:c1], nch); blen=len(blob)
        hw.put(OFF_WT, blob)
        hw.wr(REG_LAYER_TYPE,2); hw.wr(REG_CIN,4); hw.wr(REG_COUT,4); hw.wr(REG_HIN,KH); hw.wr(REG_WIN,KW)
        hw.wr(REG_N_CONFIG,1); hw.wr(REG_POC,0); hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu); hw.wr(REG_STRIDE,1); hw.wr(REG_PAD,1)
        hw.wr64(REG_IN_LO,hw.phys+OFF_ACT); hw.wr64(REG_WT_LO,hw.phys+OFF_WT); hw.wr64(REG_OUT_LO,hw.phys+OFF_OUT)
        hw.wr(REG_OUT_BYTES,64); hw.wr(REG_WT_BYTES,blen)
        def kick_tile(x0r, x0c):
            tile=np.zeros((4,KH,KW),np.int8); s=ap[c0:c1, x0r:x0r+KH, x0c:x0c+KW]; tile[:nch,:s.shape[1],:s.shape[2]]=s
            hw.put(OFF_ACT, tile.reshape(-1)); hw.kick()
        kick_tile(0,0)  # prime once per block (requant lag)
        for r in range(H):
            for cb in range(n_cb):
                x0=cb*WOUT; npos=min(WOUT,W-x0)
                kick_tile(r, x0)
                out[c0:c1, r, x0:x0+npos] = hw.get(OFF_OUT,64).reshape(4,16)[:nch,:npos]
    return out

def dwconv_layer_B0(hw, act, w, m0, shift, bias, relu=1, pad=1, to=30.0):
    """B0 dwconv: whole layer in ONE ap_start via the layer_sequencer in dw_mode (ACT indexed by
    (oc,pos) — each oc-block = 4 depthwise channels, N_ICB=1, no accumulate). Removes the per-kick
    host put/kick/get loop of dwconv_layer. w[C,3,3] (already w[:,0] from caller)."""
    C, H, W = act.shape
    ap = np.pad(act, ((0,0),(pad,pad),(pad,pad))).astype(np.int8)
    Hout, Wout = H, W                                  # dw stride1 pad1 → same size
    n_cb = (W + WOUT - 1)//WOUT
    N_OCB = (C + 3)//4; N_POS = Hout*n_cb; N_ICB = 1
    ACT_STRIDE = 256; WBB = 512                        # dw tile 4×54=216→256; blob 9·17+28=181→512 (4KB-safe)
    TILE = 4*KH*KW                                     # 216

    # ── ACT tile-stream [oc][pos][4ch×KH×KW] (channel-major: chan g at g*54; idx oc*N_POS+pos) ──
    act_s = np.zeros((N_OCB, N_POS, ACT_STRIDE), np.int8)
    for ocb in range(N_OCB):
        c0 = ocb*4; c1 = min(c0+4, C); nch = c1-c0
        for oi in range(Hout):
            for cb in range(n_cb):
                pos = oi*n_cb + cb; x0 = cb*WOUT
                tile = np.zeros((4, KH, KW), np.int8)
                s = ap[c0:c1, oi:oi+KH, x0:x0+KW]
                tile[:nch, :s.shape[1], :s.shape[2]] = s
                act_s[ocb, pos, :TILE] = tile.reshape(-1)
    hw.put(B0_ACT, act_s.reshape(-1))

    # ── WT-URAM dw blobs [oc-block] (cached: weights constant across frames) ──
    _wk = (id(w), 'dwb0', C)
    _wb = _WT_CACHE.get(_wk)
    if _wb is None:
        wt_s = np.zeros((N_OCB, N_ICB, WBB), np.uint8)
        for ocb in range(N_OCB):
            c0 = ocb*4; c1 = min(c0+4, C); nch = c1-c0
            blob = _dw_encode(w[c0:c1], m0[c0:c1], shift[c0:c1], bias[c0:c1], nch)
            wt_s[ocb, 0, :len(blob)] = np.frombuffer(blob, np.uint8)
        _wb = wt_s.reshape(-1).view(np.int8); _WT_CACHE[_wk] = _wb
    hw.put(B0_WT, _wb)

    # ── registers (LAYER_TYPE=2 dw; NOT r1_mode — dw uses the non-R1 swc path) ──
    hw.wr(REG_LAYER_TYPE,2); hw.wr(REG_CIN,4); hw.wr(REG_COUT,4); hw.wr(REG_HIN,KH); hw.wr(REG_WIN,KW)
    hw.wr(REG_N_CONFIG,1); hw.wr(REG_POC,0); hw.wr(REG_ACCU_MODE,0); hw.wr(REG_APPLY_RELU,relu); hw.wr(REG_STRIDE,1)
    hw.wr(REG_PAD, 1)                                  # dw pad (bit10 r1_mode stays 0)
    hw.wr64(REG_IN_LO, hw.phys+B0_ACT); hw.wr64(REG_WT_LO, hw.phys+B0_WT)
    hw.wr64(REG_OUT_LO, hw.phys+B0_OUT); hw.wr(REG_OUT_BYTES,64); hw.wr(REG_WT_BYTES,WBB)
    hw.wr(REG_N_OCB,N_OCB); hw.wr(REG_N_POS,N_POS); hw.wr(REG_N_ICB,N_ICB)
    hw.wr(REG_WT_BLK_BYTES,WBB); hw.wr(REG_ACT_CHUNK,ACT_STRIDE)
    hw.wr(REG_LAYER_MODE,1)
    hw.kick(to=to)
    hw.wr(REG_LAYER_MODE,0); hw.wr(REG_ACT_CHUNK,1024)

    # ── read OUT tile-stream [oc][pos][4,16] → [C,H,W] (vectorized de-interleave) ──
    out_s = hw.get(B0_OUT, N_OCB*N_POS*64).reshape(N_OCB, N_POS, 4, 16)
    out = np.ascontiguousarray(
        out_s.reshape(N_OCB, Hout, n_cb, 4, WOUT).transpose(0, 3, 1, 2, 4)
             .reshape(N_OCB*4, Hout, n_cb*WOUT)[:C, :, :W])
    return out

def dw_fast(act, w, m0, shift, bias, relu=1, pad=1, stride=1):
    """Vectorized numpy depthwise 3x3 + requant (round-shift + leaky0.1, matches rr_unit within1).
    FAST CPU dw: ~ms vs FPGA-dw ~200ms (old kernel 1% util) and quant_conv ref ~18s (unvectorized).
    w[C,3,3], act[C,H,W], m0/shift/bias[C]."""
    C,H,W=act.shape
    ap=np.pad(act,((0,0),(pad,pad),(pad,pad))).astype(np.int32)   # int32 accum (2x faster, fits 9*127^2+bias)
    Ho=(H+2*pad-3)//stride+1; Wo=(W+2*pad-3)//stride+1
    out=np.zeros((C,Ho,Wo),np.int32)
    for kh in range(3):
        for kw in range(3):
            out+=w[:,kh,kw][:,None,None].astype(np.int32)*ap[:,kh:kh+Ho*stride:stride, kw:kw+Wo*stride:stride]
    out+=bias[:,None,None].astype(np.int32)
    sh=shift[:,None,None].astype(np.int64); mm=m0[:,None,None].astype(np.int64)
    q=(out.astype(np.int64)*mm+(1<<(sh-1)))>>sh   # int64 for the m0 multiply (out*m0 overflows int32)
    if relu: q=np.where(q>=0,q,(q*205+1024)>>11)   # rr_unit leaky 0.10009 (bit-exact vs FPGA dw)
    return np.clip(q,-128,127).astype(np.int8)

def dw_fpga(hw, act, w, m0, shift, bias, relu=1, pad=1, stride=1, to=30.0):
    """On-chip depthwise (LAYER_TYPE=4): dw_layer loops C channels, 1 ap_start/layer (vs dw_fast CPU).
    act[C,H,W]; w[C,3,3]; m0/shift/bias[C]. Host pads (cheap) + lays padded channels [C,Hp,Wp] and
    per-channel 16-byte weight words [w0..w8, bias(4 LE), m0(2 LE), shift(1)] in DDR. Bit-exact dw_fast."""
    C,H,W=act.shape; Hp=H+2*pad; Wp=W+2*pad
    Ho=(Hp-3)//stride+1; Wo=(Wp-3)//stride+1
    def _algn(n): return 1024 if n<=1024 else (n+4095)//4096*4096   # 1024|4096 => no AXI 4KB cross
    acb=_algn(Hp*Wp); ocb=_algn(Ho*Wo)
    ap=np.zeros((C,Hp,Wp),np.int8); ap[:,pad:pad+H,pad:pad+W]=act.astype(np.int8)
    ap16=np.zeros((C,acb),np.int8); ap16[:,:Hp*Wp]=ap.reshape(C,Hp*Wp)
    wb=np.zeros((C,16),np.uint8)
    wb[:,:9]=np.ascontiguousarray(w.reshape(C,9).astype(np.int8)).view(np.uint8)
    b32=(bias.astype(np.int64)&0xFFFFFFFF).astype(np.uint32)
    wb[:,9]=b32&0xFF; wb[:,10]=(b32>>8)&0xFF; wb[:,11]=(b32>>16)&0xFF; wb[:,12]=(b32>>24)&0xFF
    m=m0.astype(np.uint32); wb[:,13]=m&0xFF; wb[:,14]=(m>>8)&0xFF
    wb[:,15]=shift.astype(np.uint8)&0x1F
    hw.put(B0_ACT, ap16.reshape(-1)); hw.put(B0_WT, wb.reshape(-1))
    hw.wr(REG_APPLY_RELU, 1 if relu else 0)
    hw.wr(REG_LAYER_TYPE,4); hw.wr(REG_CIN,C); hw.wr(REG_COUT,stride)
    hw.wr(REG_HIN, (Hp&0x3FF)|((Wp&0x3FF)<<10)); hw.wr(REG_WIN, (Ho&0x3FF)|((Wo&0x3FF)<<10))
    hw.wr64(REG_IN_LO,hw.phys+B0_ACT); hw.wr64(REG_WT_LO,hw.phys+B0_WT); hw.wr64(REG_OUT_LO,hw.phys+B0_OUT)
    hw.kick(to=to); hw.wr(REG_LAYER_TYPE,0)
    raw=hw.get(B0_OUT, C*ocb).reshape(C,ocb)        # aligned per-channel stride
    return raw[:,:Ho*Wo].reshape(C,Ho,Wo).astype(np.int8)

# numpy reference matching HW (round shift + leaky0.1)
def ref_conv3x3(act, w, m0, shift, bias, relu, stride=1, pad=1):
    Cout,Cin,_,_ = w.shape; _,H,W = act.shape
    ap = np.pad(act,((0,0),(pad,pad),(pad,pad))).astype(np.int64)
    Ho,Wo = (H+2*pad-3)//stride+1, (W+2*pad-3)//stride+1
    out = np.zeros((Cout,Ho,Wo),np.int64)
    for kh in range(3):
        for kw in range(3):
            patch = ap[:, kh:kh+Ho*stride:stride, kw:kw+Wo*stride:stride]
            out += np.tensordot(w[:,:,kh,kw].astype(np.int64), patch, axes=([1],[0]))
    out += bias.astype(np.int64)[:,None,None]
    sh=shift.astype(np.int64)[:,None,None]; mm=m0.astype(np.int64)[:,None,None]
    q=(out*mm + (1<<(sh-1)))>>sh
    if relu: q=np.where(q>=0,q,np.round(0.1*q).astype(np.int64))
    return np.clip(q,-128,127).astype(np.int8)

def conv1x1_systolic(hw, act, w1, m0, shift, bias, relu=1, to=30.0, _skip_get=False):
    """R4 SYSTOLIC conv1x1 (LAYER_TYPE=3): banked on-chip fmap + 256-MAC array, ~86% util.
    act[Cin,H,W]; w1[Cout,Cin,1,1] or [Cout,Cin]; m0/shift/bias[Cout]. Params travel in the WT DMA
    ([wmem words][param words], 2 Cout per 128-bit param word). OUT<=1024B => Cout*ceil(HW/16)<=64."""
    N=16; _worig=id(w1)
    w1=w1.reshape(w1.shape[0],w1.shape[1])
    Cout,Cin=w1.shape; _,H,W=act.shape; HWn=H*W
    WB=(HWn+N-1)//N; N_OCB=(Cout+N-1)//N; nwmem=N_OCB*Cin; npar=(Cout+1)//2
    # amem[ci*WB+p][lane]=A[ci][p*16+lane]  (per-frame). HW mult of 16 (large-spatial layers) => the
    # raw activation IS the bank layout (16 consecutive hw per 128-bit word) => skip zeros+pad copy.
    _ta=time.time()
    af=np.ascontiguousarray(act.reshape(Cin,HWn))   # int8 already; contiguous for the put
    if HWn==WB*N:
        amem=af
    else:
        amem=np.zeros((Cin,WB*N),np.int8); amem[:,:HWn]=af
    _LP['amem']=_LP.get('amem',0)+time.time()-_ta
    # wt blob = [wmem][pmem] cached by weight identity (weights fixed across frames)
    _wk=(_worig,'sys',Cout,Cin)
    wt=_WT_CACHE.get(_wk)
    if wt is None:
        wp=np.zeros((N_OCB*N,Cin),np.int8); wp[:Cout,:]=w1.astype(np.int8)
        wmem=np.ascontiguousarray(wp.reshape(N_OCB,N,Cin).transpose(0,2,1)).reshape(nwmem,N)
        # pmem vectorized: half=bias<<24|shift<<16|m0, 2 Cout per 128-bit word
        half=((bias.astype(np.int64)&0xFFFFFFFF).astype(np.uint64)<<24)|((shift.astype(np.uint64)&0x1F)<<16)|(m0.astype(np.uint64)&0xFFFF)
        hpad=np.zeros(npar*2,np.uint64); hpad[:Cout]=half
        pmem=hpad.view(np.uint8).reshape(npar,N)
        wt=np.concatenate([wmem.reshape(-1).view(np.int8), pmem.reshape(-1).view(np.int8)])
        _WT_CACHE[_wk]=wt
    hw.put(B0_ACT, amem.reshape(-1)); hw.put(B0_WT, wt)
    hw.wr(REG_APPLY_RELU, 1 if relu else 0)   # minimal top: per-layer relu (no-relu int8 convs exist)
    hw.wr(REG_LAYER_TYPE,3); hw.wr(REG_CIN,Cin); hw.wr(REG_COUT,Cout); hw.wr(REG_HIN,HWn); hw.wr(REG_WIN,1)
    hw.wr64(REG_IN_LO,hw.phys+B0_ACT); hw.wr64(REG_WT_LO,hw.phys+B0_WT); hw.wr64(REG_OUT_LO,hw.phys+B0_OUT)
    hw.wr(REG_WT_BYTES,(nwmem+npar)*16); hw.wr(REG_OUT_BYTES,Cout*WB*16)
    _kick(hw,'c1',to); hw.wr(REG_LAYER_TYPE,0)
    # Note: _T2_STATE['c1'] is set by the tiled wrapper after all tiles.
    if _skip_get:
        # C1→C1 chain producer: skip hw.get() — data stays in c1 omem for c1_copy.
        # Return zeros placeholder with correct shape (consumer uses LT7, ignores act arg).
        return np.zeros((Cout,WB*N),np.int8).reshape(Cout,WB*N)[:,:HWn].reshape(Cout,H,W)
    _td=time.time()
    _o=hw.get(B0_OUT,Cout*WB*16).reshape(Cout,WB,N).reshape(Cout,WB*N)[:,:HWn].reshape(Cout,H,W)
    _LP['deint']=_LP.get('deint',0)+time.time()-_td
    return _o

SYS_ADEPTH=16384; SYS_ODEPTH=8192   # on-chip URAM amem/wmem depth, BRAM omem depth
def conv1x1_systolic_tiled(hw, act, w1, m0, shift, bias, relu=1, skip_get=False):
    """conv1x1 systolic with HW-tiling (point-wise: amem=Cin*WB<=ADEPTH, omem=Cout*WB<=ODEPTH) AND
    COUT-tiling (wmem=ceil(Cout/16)*Cin+params must fit the 14-bit wt counter / ADEPTH; output
    channels independent, no psum accumulation). Returns None only if Cin>ADEPTH."""
    N=16; Cout=w1.shape[0]; Cin=w1.shape[1]; _,H,W=act.shape; HWn=H*W
    if Cin > SYS_ADEPTH: return None
    # Cout-tile so wmem (ocb*Cin + params) fits — splits the big-channel layers (512x512, 1024x512)
    ocb=max(1,(SYS_ADEPTH-512)//Cin); cchunk=ocb*N
    if Cout > cchunk:
        outs=[]; k0=id(w1)
        for co in range(0, Cout, cchunk):
            ce=min(co+cchunk, Cout); ck=('c1cout',k0,co)
            if ck not in _W3CACHE: _W3CACHE[ck]=np.ascontiguousarray(w1[co:ce])
            o=conv1x1_systolic_tiled(hw, act, _W3CACHE[ck], m0[co:ce], shift[co:ce], bias[co:ce], relu=relu)
            if o is None: return None
            outs.append(o)
        return np.concatenate(outs, axis=0)
    af=act.reshape(Cin,HWn)
    cmax=min(1008, (SYS_ADEPTH*N)//Cin, (SYS_ODEPTH*N)//Cout); cmax=max(N,(cmax//N)*N)
    outs=[]
    tiles=list(range(0, HWn, cmax))
    for i,s in enumerate(tiles):
        e=min(s+cmax, HWn)
        is_last = (i == len(tiles)-1)
        # skip_get only on last (and only) tile of a chain producer (single-tile guaranteed by caller)
        o=conv1x1_systolic(hw, af[:,s:e].reshape(Cin,1,e-s), w1, m0, shift, bias, relu=relu,
                           _skip_get=(skip_get and is_last))
        outs.append(o.reshape(Cout,e-s))
    result = np.concatenate(outs, axis=1).reshape(Cout,H,W)
    if HWn <= cmax:   # single tile: c1 omem has full layer output
        _T2_STATE['c1'] = (Cout, H, W)
    else:
        _T2_STATE['c1'] = None
    return result

_W3CACHE={}
_LP={}   # conv-layout numpy profiler (gated by LPROF env)
_KT={}   # per-type HW-kick profiler (gated by KPROF env): c1=conv1x1, c3=conv3x3
if os.environ.get('LPROF'):
    import atexit
    atexit.register(lambda: print("    LPROF conv-layout numpy: " +
        " ".join(f"{k}={v:.2f}s" for k,v in sorted(_LP.items()))))
if os.environ.get('KPROF'):
    import atexit
    atexit.register(lambda: print("    KPROF HW-kick per type: " +
        " ".join(f"{k}={v:.2f}s" for k,v in sorted(_KT.items()))))
def _kick(hw, tag, to=30.0):
    if os.environ.get('KPROF'):
        _t=time.time(); hw.kick(to=to); _KT[tag]=_KT.get(tag,0)+time.time()-_t
    else: hw.kick(to=to)

# ── Network Sequencer (LT14): 2 conv1x1 chained, ONE ap_start, no host round-trip ──────────────
def _build_c1_amem(act):
    Cin,H,W=act.shape; HWn=H*W; N=16; WB=(HWn+N-1)//N
    af=np.ascontiguousarray(act.reshape(Cin,HWn))
    if HWn==WB*N: amem=af
    else: amem=np.zeros((Cin,WB*N),np.int8); amem[:,:HWn]=af
    return amem.reshape(-1), WB

def _build_c1_wt_blob(w1,m0,shift,bias):
    N=16; Cout,Cin=w1.shape; N_OCB=(Cout+N-1)//N; nwmem=N_OCB*Cin; npar=(Cout+1)//2
    wp=np.zeros((N_OCB*N,Cin),np.int8); wp[:Cout,:]=w1.astype(np.int8)
    wmem=np.ascontiguousarray(wp.reshape(N_OCB,N,Cin).transpose(0,2,1)).reshape(nwmem,N)
    half=((bias.astype(np.int64)&0xFFFFFFFF).astype(np.uint64)<<24)|((shift.astype(np.uint64)&0x1F)<<16)|(m0.astype(np.uint64)&0xFFFF)
    hpad=np.zeros(npar*2,np.uint64); hpad[:Cout]=half
    pmem=hpad.view(np.uint8).reshape(npar,N)
    return np.concatenate([wmem.reshape(-1).view(np.int8),pmem.reshape(-1).view(np.int8)]), nwmem, npar

def net_seq_run(hw, act0, layers, relu=1, to=30.0):
    """LT14 DESCRIPTOR mode (Rev132): chain N conv1x1 layers under ONE ap_start. layer i+1 act =
    layer i out (DDR roundtrip, host never touches intermediate). layers=[(w,m0,sh,b),...]. Same HW.
    Config passed via DDR descriptor table (64B/layer) — desc_base=REG_IN_LO, n_layers=REG_CIN. No
    new AXI registers (the register-based PoC hit the AXI address wall, see addr-wall memo).
    Descriptor[i] words: 0=cin 1=cout 2=hw 3=act_bytes 4=wt_bytes 5=out_bytes
                         6/7=act_addr lo/hi  8/9=wt_addr lo/hi  10/11=out_addr lo/hi."""
    N=16; _,H,W=act0.shape; HWn=H*W; WB=(HWn+N-1)//N
    NL=len(layers)
    NS_DESC=0x0700000                       # descriptor table (in ACT region, past layer0 amem)
    NS_ACT0=B0_ACT                          # layer-0 activation
    WT_STRIDE=0x1000000; OUT_STRIDE=0x400000
    wt_off =[B0_WT + i*WT_STRIDE  for i in range(NL)]
    out_off=[B0_OUT+ i*OUT_STRIDE for i in range(NL)]
    desc=np.zeros((NL,16),np.uint32)
    for i,(w,m0,sh,b) in enumerate(layers):
        w=w.reshape(w.shape[0],w.shape[1]); Cout,Cin=w.shape
        if i>0:
            assert Cin==layers[i-1][0].shape[0], f"chain L{i}: Cin({Cin})!=prev Cout"
        wt,nwm,npar=_build_c1_wt_blob(w,m0,sh,b)
        hw.put(wt_off[i], wt)
        a_off = NS_ACT0 if i==0 else out_off[i-1]
        aa=hw.phys+a_off; wa=hw.phys+wt_off[i]; oa=hw.phys+out_off[i]
        desc[i,0]=Cin; desc[i,1]=Cout; desc[i,2]=HWn
        desc[i,3]=Cin*WB*16; desc[i,4]=(nwm+npar)*16; desc[i,5]=Cout*WB*16
        desc[i,6]=aa&0xFFFFFFFF; desc[i,7]=(aa>>32)&0xFFFFFFFF
        desc[i,8]=wa&0xFFFFFFFF; desc[i,9]=(wa>>32)&0xFFFFFFFF
        desc[i,10]=oa&0xFFFFFFFF; desc[i,11]=(oa>>32)&0xFFFFFFFF
    amem0,_=_build_c1_amem(act0)
    hw.put(NS_ACT0, amem0)
    hw.put(NS_DESC, np.ascontiguousarray(desc).reshape(-1).view(np.int8))
    hw.wr(REG_APPLY_RELU,relu)
    hw.wr64(REG_IN_LO, hw.phys+NS_DESC)     # desc_base (reused register)
    hw.wr(REG_CIN, NL)                       # n_layers (reused register)
    hw.wr(REG_LAYER_TYPE,14); hw.kick(to=to); hw.wr(REG_LAYER_TYPE,0)
    CoutL=layers[-1][0].shape[0]
    return hw.get(out_off[NL-1], CoutL*WB*16).reshape(CoutL,WB,N).reshape(CoutL,WB*N)[:,:HWn].reshape(CoutL,H,W)

# ── T2 On-chip dataflow (Phase 2): cross-engine copies without DDR round-trip ─────────────────
# RTL: LAYER_TYPE=6 (xcopy c3→c1 amem), LT7 (c1 skip act), LT8 (pad_rebank c1→c3 amem),
#       LT9 (c3 skip act), LT10 (dw reads c1 omem, dwoc mode). All board-verified (rev109).
# Usage: after a C3/C1 kick, call the chain helper before next layer instead of hw.get()+hw.put().

# Last C1/C3 kick descriptor: tracks on-chip URAM state for chaining decisions.
_T2_STATE = {'c1': None, 'c3': None}   # None or (Cout, H, W)

def c1_copy_kick(hw, Cout, H, W, to=0.5):
    """LT13 (c1_copy): copy c1 omem → c1 amem on-chip (no DDR). Phase 2 on-chip dataflow.
    Called BETWEEN two consecutive conv1x1 layers (producer → consumer) to avoid DDR round-trip.
    Duration: ~Cout*ceil(HW/16) clock cycles = µs range. Much faster than DDR DMA.
    After this: consumer can kick with LT7 (is_c1na, skip ACT DMA)."""
    N=16; WB=(H*W+N-1)//N
    hw.wr(REG_LAYER_TYPE, 13)   # LT13: is_c1c
    hw.wr(REG_COUT, Cout); hw.wr(REG_HIN, H*W); hw.wr(REG_WIN, 1)  # reuse to compute n_words=Cout*WB
    hw.kick(to=to); hw.wr(REG_LAYER_TYPE, 0)

def t2_xcopy(hw, Cout, Ho, Wo, pad=1):
    """T2-2: copy c3 omem → c1 amem in-place (rebank). LAYER_TYPE=6.
    Must match the shape of the C3 output that just computed into c3 omem.
    ~50-200µs on-chip BRAM copy (no DDR). After this, C1 can kick with LT7."""
    Hp = Ho+2*pad; Wp = Wo+2*pad
    hw.wr(REG_LAYER_TYPE, 6); hw.wr(REG_COUT, Cout)
    hw.wr(REG_HIN, (Hp & 0x3FF) | ((Wp & 0x3FF)<<10))
    hw.wr(REG_WIN, (Ho & 0x3FF) | ((Wo & 0x3FF)<<10))
    hw.kick(to=2.0); hw.wr(REG_LAYER_TYPE, 0)

def t2_pad_rebank(hw, Cout, Ho, Wo):
    """T2-1: copy c1 omem → c3 amem PADDED (pad_rebank). LAYER_TYPE=8.
    Must match the C1 output shape. After this, C3 can kick with LT9."""
    hw.wr(REG_LAYER_TYPE, 8); hw.wr(REG_COUT, Cout)
    hw.wr(REG_WIN, (Ho & 0x3FF) | ((Wo & 0x3FF)<<10))
    hw.kick(to=2.0); hw.wr(REG_LAYER_TYPE, 0)

def conv1x1_systolic_skip_act(hw, Cin, Cout, H, W, wt_blob, relu, to=30.0):
    """Conv1x1 systolic with LT7 (c1na): skip ACT DMA, read pre-filled c1 amem directly.
    wt_blob: pre-built [wmem][pmem] blob (16-byte words). Returns INT8 [Cout,H,W]."""
    N=16; HWn=H*W; WB=(HWn+N-1)//N
    hw.put(B0_WT, wt_blob)
    hw.wr(REG_APPLY_RELU, int(relu))
    hw.wr(REG_LAYER_TYPE, 7)   # LT7: is_c1na — skip ACT DMA, read amem from xcopy
    hw.wr(REG_CIN, Cin); hw.wr(REG_COUT, Cout); hw.wr(REG_HIN, HWn); hw.wr(REG_WIN, 1)
    hw.wr64(REG_WT_LO, hw.phys+B0_WT); hw.wr64(REG_OUT_LO, hw.phys+B0_OUT)
    N_OCB=(Cout+N-1)//N; nwmem=N_OCB*Cin; npar=(Cout+1)//2
    hw.wr(REG_WT_BYTES, (nwmem+npar)*16); hw.wr(REG_OUT_BYTES, Cout*WB*16)
    _kick(hw, 'c1t2', to); hw.wr(REG_LAYER_TYPE, 0)
    _T2_STATE['c1'] = (Cout, H, W)
    _td=time.time()
    out=hw.get(B0_OUT, Cout*WB*16).reshape(Cout,WB,N).reshape(Cout,WB*N)[:,:HWn].reshape(Cout,H,W)
    _LP['deint']=_LP.get('deint',0)+time.time()-_td
    return out

def dw_fpga_from_c1omem(hw, C, Ho, Wo, stride, w, m0, shift, bias, relu=1, to=30.0):
    """T2-3: DW using LAYER_TYPE=10 (dwoc) — reads from c1 omem directly (no ACT DMA).
    C1 output must already be in c1 omem from previous C1 kick (tracked by _T2_STATE['c1']).
    Saves: host hw.put(act) + FPGA internal ACT DMA for the DW layer."""
    Hp=Ho+stride; Wp=Wo+stride   # not used for dwoc (no padding needed, reads c1 omem)
    WBsrc=(Ho*Wo+15)//16          # ceil(Ho*Wo/16): words/channel in c1 omem
    acb=(Ho*Wo+1023)//1024*1024   # output stride
    wb=np.zeros((C,16),np.uint8)
    wb[:,:9]=np.ascontiguousarray(w.reshape(C,9).astype(np.int8)).view(np.uint8)
    b32=(bias.astype(np.int64)&0xFFFFFFFF).astype(np.uint32)
    wb[:,9]=b32&0xFF; wb[:,10]=(b32>>8)&0xFF; wb[:,11]=(b32>>16)&0xFF; wb[:,12]=(b32>>24)&0xFF
    m=m0.astype(np.uint32); wb[:,13]=m&0xFF; wb[:,14]=(m>>8)&0xFF
    wb[:,15]=shift.astype(np.uint8)&0x1F
    hw.put(B0_WT, wb.reshape(-1))
    hw.wr(REG_APPLY_RELU, 1 if relu else 0)
    hw.wr(REG_LAYER_TYPE, 10)  # LT10: is_dwoc — DW reads c1 omem, skip act DMA
    hw.wr(REG_CIN, C); hw.wr(REG_COUT, stride)   # REG_COUT[1:0] = stride for DW
    hw.wr(REG_HIN, (Ho & 0x3FF)|((Wo & 0x3FF)<<10))
    hw.wr(REG_WIN, (Ho & 0x3FF)|((Wo & 0x3FF)<<10))
    hw.wr64(REG_WT_LO, hw.phys+B0_WT); hw.wr64(REG_OUT_LO, hw.phys+B0_OUT)
    _kick(hw, 'dwt2', to); hw.wr(REG_LAYER_TYPE, 0)
    _T2_STATE['c1'] = None   # c1 omem consumed
    raw=hw.get(B0_OUT, C*acb).reshape(C,acb)
    return raw[:,:Ho*Wo].reshape(C,Ho,Wo).astype(np.int8)

def conv3x3_systolic(hw, act, w, m0, shift, bias, relu=1, pad=1, stride=1):
    """conv3x3 via im2col -> systolic conv1x1, Cin'=9*Cin virtual channels. W'[co][ci*9+kh*3+kw]
    =W[co][ci][kh][kw]; A'[ci*9+kh*3+kw][ho][wo]=A_pad[ci][ho*s+kh][wo*s+kw]. The 9x channel inflation
    makes wmem (=ceil(Cout/16)*9*Cin) too big => COUT-TILING: split Cout into blocks whose wmem fits
    URAM (output channels are independent, NO psum accumulation needed). HW-tiling (inside the conv1x1
    helper) keeps amem in range. Host-only — no RTL keep_psum."""
    N=16; Cout,Cin,_,_=w.shape; _,H,W=act.shape; Cinp=Cin*9
    Ho=(H+2*pad-3)//stride+1; Wo=(W+2*pad-3)//stride+1
    ocb=max(1,(SYS_ADEPTH-256)//Cinp)                     # co-blocks (16 each) whose wmem fits
    cchunk=ocb*N
    if Cinp > SYS_ADEPTH: return None                      # even 1 co-block won't fit -> fallback
    # im2col A' once (shared by all Cout-chunks)
    _t0=time.time()
    ap=np.pad(act,((0,0),(pad,pad),(pad,pad))).astype(np.int8)
    Ap=np.zeros((Cin,9,Ho,Wo),np.int8)
    for kh in range(3):
        for kw in range(3): Ap[:,kh*3+kw]=ap[:,kh:kh+Ho*stride:stride, kw:kw+Wo*stride:stride]
    Ap=Ap.reshape(Cinp,Ho,Wo)
    _LP['im2col']=_LP.get('im2col',0)+time.time()-_t0
    if id(w) not in _W3CACHE: _W3CACHE[id(w)]=np.ascontiguousarray(w.reshape(Cout,Cinp))
    Wf=_W3CACHE[id(w)]
    outs=[]
    for co in range(0, Cout, cchunk):
        ce=min(co+cchunk, Cout); ck=(id(w),co)
        if ck not in _W3CACHE: _W3CACHE[ck]=np.ascontiguousarray(Wf[co:ce])   # stable id -> wmem cache
        o=conv1x1_systolic_tiled(hw, Ap, _W3CACHE[ck], m0[co:ce], shift[co:ce], bias[co:ce], relu=relu)
        if o is None: return None
        outs.append(o)
    return np.concatenate(outs, axis=0)

def ref_conv3x3_rr(act, w, m0, sh, bias, relu, pad=1, stride=1):
    """conv3x3 reference with rr_unit-exact requant (round-half-up shift + leaky (q*205+1024)>>11)."""
    Cout,Cin,_,_=w.shape; _,H,W=act.shape
    ap=np.pad(act,((0,0),(pad,pad),(pad,pad))).astype(np.int64)
    Ho=(H+2*pad-3)//stride+1; Wo=(W+2*pad-3)//stride+1
    out=np.zeros((Cout,Ho,Wo),np.int64)
    for kh in range(3):
        for kw in range(3):
            patch=ap[:,kh:kh+Ho*stride:stride,kw:kw+Wo*stride:stride]
            out+=np.tensordot(w[:,:,kh,kw].astype(np.int64),patch,axes=([1],[0]))
    out+=bias[:,None,None].astype(np.int64)
    s=sh[:,None,None].astype(np.int64); q=(out*m0[:,None,None].astype(np.int64)+(1<<(s-1)))>>s
    if relu: q=np.where(q>=0,q,(q*205+1024)>>11)
    return np.clip(q,-128,127).astype(np.int8)

SYS3_AD=16384; SYS3_OD=8192; SYS3_WD=4096        # conv3x3 amem/omem/wmem (WD->4096 single-URAM, H3 200MHz; model max Cin9=2304)
def conv3x3_onchip(hw, act, w, m0, shift, bias, relu=1, pad=1, stride=1, to=30.0):
    """conv3x3 via ON-CHIP im2col (LAYER_TYPE=5). Host puts RAW padded act (no 9x A'); engine windows
    the 3x3 taps. Cout-tiled to fit wmem/omem; ROW-BAND tiled so the raw act fits amem (H1: the model's
    big high-res conv3x3 used to fall back to host im2col — now they run on-chip). Returns [Cout,Ho,Wo].
    None=fallback (stride!=1, or Cin so big even a 3-row band / 1 co-block won't fit)."""
    if stride!=1: return None
    Cout,Cin,_,_=w.shape; _,H,W=act.shape; N=16; Cin9=Cin*9
    Hp,Wp=H+2*pad,W+2*pad; Ho,Wo=Hp-2,Wp-2; WBrow=(Wo+N-1)//N
    if Cin9 > SYS3_WD: return None                  # weights of even 1 co-block won't fit wmem
    # ROW-BAND tiling: max band height Hb bounded by BOTH on-chip memories —
    #   amem (raw act):  Cin*ceil((Hb+2)*Wp/16) <= SYS3_AD  -> Hb_amem
    #   omem (output):   one co-block (16) * Hb * WBrow <= SYS3_OD  -> Hb_omem
    # (band output rows [r0,r0+Hb) use padded rows [r0,r0+Hb+2).)
    maxw = SYS3_AD // Cin                            # max amem words per channel
    Hb_amem = (maxw*N)//Wp - 2 if maxw >= 1 else 0   # rows so (Hb+2)*Wp <= maxw*16
    Hb_omem = SYS3_OD // (N*WBrow)                    # rows so 16*Hb*WBrow <= SYS3_OD
    Hbmax = min(Hb_amem, Hb_omem, Ho)
    if Hbmax < 1: return None                        # not even a 1-row band fits -> host fallback
    # cchunk fixed by the LARGEST band so co-chunk boundaries (and the WT cache key) match every band
    ocb_wm = max(1, SYS3_WD//Cin9); ocb_om = max(1, SYS3_OD//(N*Hbmax*WBrow))
    cchunk = min(ocb_wm, ocb_om)*N
    if cchunk < N: return None                       # even 16 Cout won't fit omem -> fallback
    wkey=('c3o',id(w))
    if wkey not in _W3CACHE: _W3CACHE[wkey]=np.ascontiguousarray(w.reshape(Cout,Cin9))
    Wf=_W3CACHE[wkey]
    ap_full=np.pad(act,((0,0),(pad,pad),(pad,pad))).astype(np.int8)   # [Cin,Hp,Wp]
    bands=[]
    for r0 in range(0, Ho, Hbmax):
        Hb=min(Hbmax, Ho-r0); Hps=Hb+2; Hos=Hb       # padded rows fed / output rows produced
        WBin=(Hps*Wp+N-1)//N
        abank=np.zeros((Cin,WBin*N),np.int8)
        abank[:,:Hps*Wp]=ap_full[:,r0:r0+Hps,:].reshape(Cin,Hps*Wp)
        hw.put(B0_ACT, abank.reshape(-1))
        cols=[]
        for co in range(0,Cout,cchunk):
            ce=min(co+cchunk,Cout); cc=ce-co; N_OCB=(cc+N-1)//N
            wk=('c3owm',id(w),co)
            wt=_WT_CACHE.get(wk)
            if wt is None:
                wp=np.zeros((N_OCB*N,Cin9),np.int8); wp[:cc]=Wf[co:ce]
                wmem=np.ascontiguousarray(wp.reshape(N_OCB,N,Cin9).transpose(0,2,1)).reshape(N_OCB*Cin9,N)
                npar=(cc+1)//2
                half=((bias[co:ce].astype(np.int64)&0xFFFFFFFF).astype(np.uint64)<<24)|((shift[co:ce].astype(np.uint64)&0x1F)<<16)|(m0[co:ce].astype(np.uint64)&0xFFFF)
                hp=np.zeros(npar*2,np.uint64); hp[:cc]=half
                pmem=hp.view(np.uint8).reshape(npar,N)
                wt=np.concatenate([wmem.reshape(-1).view(np.int8),pmem.reshape(-1).view(np.int8)])
                _WT_CACHE[wk]=wt
            hw.put(B0_WT, wt)
            nwmem=N_OCB*Cin9; npar=(cc+1)//2
            hw.wr(REG_APPLY_RELU,1 if relu else 0)
            hw.wr(REG_LAYER_TYPE,5); hw.wr(REG_CIN,Cin); hw.wr(REG_COUT,cc)
            hw.wr(REG_HIN,(Hps&0x3FF)|((Wp&0x3FF)<<10)); hw.wr(REG_WIN,(Hos&0x3FF)|((Wo&0x3FF)<<10))
            hw.wr64(REG_IN_LO,hw.phys+B0_ACT); hw.wr64(REG_WT_LO,hw.phys+B0_WT); hw.wr64(REG_OUT_LO,hw.phys+B0_OUT)
            hw.wr(REG_WT_BYTES,(nwmem+npar)*16); hw.wr(REG_OUT_BYTES,cc*Hos*WBrow*16)
            _kick(hw,'c3',to); hw.wr(REG_LAYER_TYPE,0)
            o=hw.get(B0_OUT,cc*Hos*WBrow*16).reshape(cc,Hos,WBrow*16)[:,:,:Wo].astype(np.int8)
            cols.append(o)
        bands.append(np.concatenate(cols,axis=0))    # [Cout, Hos, Wo] for this band
    result = np.concatenate(bands,axis=1)            # stack bands along H -> [Cout, Ho, Wo]
    # Track for T2-2 (Phase 2): last C3 output shape in c3 omem for C3→C1 chain via xcopy.
    # Only valid for single-band full-layer (n_bands=1); multi-band xcopy not supported.
    if len(bands) == 1:
        _T2_STATE['c3'] = (Cout, Ho, Wo)
    else:
        _T2_STATE['c3'] = None
    return result

# ── conv3x3 on-chip layer sequencer (LAYER_TYPE=5 + LAYER_MODE=1) ───────────────
# One ap_start runs all (band × Cout-chunk) loops on-chip via conv3x3_layer_seq RTL.
# Host pre-stages padded-act bands + wt chunks in DDR; sequencer reads and kicks internally.
_C3SEQ_WT_CACHE = {}   # wt+param chunks keyed by (w_id, co): persist across frames (warm reuse)

def conv3x3_onchip_seq(hw, act, w, m0, shift, bias, relu=1, pad=1, stride=1, to=60.0):
    """conv3x3 on-chip layer sequencer (LAYER_TYPE=5+LAYER_MODE=1): single ap_start runs all
    Cout-chunk iterations on-chip. Requires n_bands=1 (whole image fits in amem).
    Uses uniform REG_COUT=cchunk so the sequencer can use fixed register config.
    Last (partial) Cout-chunk is zero-padded to cchunk — we discard the extra channels on collection."""
    if stride != 1: return None
    Cout, Cin, _, _ = w.shape; _, H, W = act.shape; N = 16
    Hp, Wp = H+2*pad, W+2*pad; Ho, Wo = Hp-2, Wp-2; WBrow = (Wo+N-1)//N
    if Cin*9 > SYS3_WD: return None
    maxw = SYS3_AD // Cin
    Hb_amem = (maxw*N)//Wp - 2 if maxw >= 1 else 0
    Hb_omem = SYS3_OD // (N*WBrow)
    Hbmax = min(Hb_amem, Hb_omem, Ho)
    if Hbmax < 1: return None
    if Hbmax < Ho: return None   # only support n_bands=1 (multi-band → fall back to conv3x3_onchip)
    ocb_wm = max(1, SYS3_WD//(Cin*9)); ocb_om = max(1, SYS3_OD//(N*Ho*WBrow))
    cchunk = min(ocb_wm, ocb_om)*N
    if cchunk < N: return None

    cout_chunks = list(range(0, Cout, cchunk)); n_cout = len(cout_chunks)

    # N_OCB_u: UNIFORM oc-block count per chunk (cchunk/N) — all chunks use same value so
    # REG_COUT=cchunk and wt param offset are consistent across chunks (last chunk zero-padded).
    N_OCB_u = cchunk // N
    npar_u   = (cchunk + 1) // 2   # uniform param words per chunk

    # ── Pre-stage weight chunks in DDR (cached; uniform size = wt_chunk_bytes_u) ──
    w_id = id(w)
    wt_chunk_bytes_u = (N_OCB_u * (Cin*9) + npar_u) * N
    if ('c3s_wf', w_id) not in _C3SEQ_WT_CACHE:
        _C3SEQ_WT_CACHE[('c3s_wf', w_id)] = np.ascontiguousarray(w.reshape(Cout, Cin*9))
    Wf = _C3SEQ_WT_CACHE[('c3s_wf', w_id)]

    wt_blob = np.zeros(n_cout * wt_chunk_bytes_u, dtype=np.int8)
    for ci, co in enumerate(cout_chunks):
        ce = min(co+cchunk, Cout); cc = ce-co
        ck = ('c3su', w_id, co)
        if ck not in _C3SEQ_WT_CACHE:
            wp = np.zeros((N_OCB_u*N, Cin*9), np.int8); wp[:cc] = Wf[co:ce]
            wmem = np.ascontiguousarray(wp.reshape(N_OCB_u, N, Cin*9).transpose(0,2,1)).reshape(N_OCB_u*Cin*9, N)
            # Params: uniform npar_u words, valid only for first cc channels (rest zero → m0=0→out=0)
            half_u = np.zeros(npar_u*2, np.uint64)
            half_valid = ((bias[co:ce].astype(np.int64)&0xFFFFFFFF).astype(np.uint64)<<24) | \
                         ((shift[co:ce].astype(np.uint64)&0x1F)<<16) | (m0[co:ce].astype(np.uint64)&0xFFFF)
            half_u[:cc] = half_valid
            pmem = half_u.view(np.uint8).reshape(npar_u, N)
            _C3SEQ_WT_CACHE[ck] = np.concatenate([wmem.reshape(-1).view(np.int8), pmem.reshape(-1).view(np.int8)])
        wt_blob[ci*wt_chunk_bytes_u : (ci+1)*wt_chunk_bytes_u] = _C3SEQ_WT_CACHE[ck]
    hw.put(B0_WT, wt_blob)

    # ── Pre-stage act (single band since n_bands=1) ──
    Hps = Ho+2; WBin = (Hps*Wp + N-1)//N
    raw_bytes = Cin * WBin * N
    act_band_bytes = ((raw_bytes + 4095)//4096)*4096   # 4KB-align
    ap_full = np.pad(act, ((0,0),(pad,pad),(pad,pad))).astype(np.int8)
    abank = np.zeros((Cin, WBin*N), np.int8); abank[:, :Hps*Wp] = ap_full.reshape(Cin, Hps*Wp)
    act_blob = np.zeros(act_band_bytes, dtype=np.int8); act_blob[:raw_bytes] = abank.reshape(-1)
    hw.put(B0_ACT, act_blob)

    # Output: uniform cchunk channels × Ho rows × WBrow blocks × N (same for all chunks)
    out_chunk_bytes = cchunk * Ho * WBrow * N

    # ── Configure sequencer registers ──
    hw.wr(REG_LAYER_TYPE, 5)
    hw.wr(REG_CIN, Cin); hw.wr(REG_COUT, cchunk)   # UNIFORM cchunk (not Cout!)
    hw.wr(REG_APPLY_RELU, 1 if relu else 0)
    hw.wr64(REG_IN_LO,  hw.phys + B0_ACT)
    hw.wr64(REG_WT_LO,  hw.phys + B0_WT)
    hw.wr64(REG_OUT_LO, hw.phys + B0_OUT)
    hw.wr(REG_N_POS,  1)           # n_bands = 1
    hw.wr(REG_N_OCB,  n_cout)      # Cout-chunk count
    hw.wr(REG_ACT_CHUNK, act_band_bytes)     # act band bytes (also stride; only 1 band)
    hw.wr(REG_WT_BYTES,  wt_chunk_bytes_u)  # uniform wt chunk bytes
    hw.wr(REG_OUT_BYTES, out_chunk_bytes)   # uniform output chunk bytes
    hw.wr(REG_HIN, (Hps & 0x3FF) | ((Wp & 0x3FF)<<10))
    hw.wr(REG_WIN, (Ho  & 0x3FF) | ((Wo & 0x3FF)<<10))
    hw.wr(REG_LAYER_MODE, 1)
    hw.kick(to=to)
    hw.wr(REG_LAYER_MODE, 0); hw.wr(REG_LAYER_TYPE, 0)

    # ── Collect output: n_cout uniform chunks in B0_OUT, each [cchunk, Ho, WBrow*N] ──
    ob_all = hw.get(B0_OUT, n_cout * out_chunk_bytes)
    cols = []
    for ci, co in enumerate(cout_chunks):
        cc = min(co+cchunk, Cout) - co   # valid channels in this chunk
        raw = ob_all[ci*out_chunk_bytes : (ci+1)*out_chunk_bytes]
        # Engine writes [cchunk, Ho, WBrow*N]; take first cc channels only
        o = raw.reshape(cchunk, Ho, WBrow*N)[:cc, :, :Wo].astype(np.int8)
        cols.append(o)
    return np.concatenate(cols, axis=0)   # [Cout, Ho, Wo]

if __name__=="__main__":
    import sys
    hw=HW()
    if "--systolictest" in sys.argv:
        for (Cin,Cout,H,W) in [(16,8,4,4),(32,8,4,4),(64,16,4,4),(256,8,4,4),(512,8,4,4)]:
            rng=np.random.default_rng(Cin*7+Cout)
            act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8)
            w1=rng.integers(-2,3,(Cout,Cin,1,1)).astype(np.int8)
            m0=rng.integers(0x2000,0x7000,Cout).astype(np.uint16); shift=rng.integers(14,18,Cout).astype(np.uint8)
            bias=rng.integers(-40,40,Cout).astype(np.int32)
            w3=np.zeros((Cout,Cin,3,3),np.int8); w3[:,:,1,1]=w1[:,:,0,0]
            ref=ref_conv3x3(act,w3,m0,shift,bias,relu=1,pad=1)
            r =conv1x1_systolic(hw,act,w1,m0,shift,bias,relu=1)
            d=np.abs(r.astype(np.int16)-ref.astype(np.int16))
            print(f"[SYSTOLIC 1x1] Cin={Cin:3d} Cout={Cout:2d} {H}x{W} out{r.shape}: within1={100*np.mean(d<=1):5.1f}% max={d.max():3d}")
        sys.exit(0)
    if "--dwfpgatest" in sys.argv:   # on-chip dw_engine (LAYER_TYPE=4) vs dw_fast reference
        for (C,H,W,st) in [(4,8,8,1),(8,8,8,2),(16,26,26,1),(32,52,52,1),(64,13,13,1),(32,104,104,1)]:
            rng=np.random.default_rng(C*3+H)
            act=rng.integers(-100,100,(C,H,W)).astype(np.int8)
            w=rng.integers(-30,30,(C,3,3)).astype(np.int8)
            m0=rng.integers(0x4000,0x9000,C).astype(np.uint16); shift=rng.integers(16,20,C).astype(np.uint8)
            bias=rng.integers(-2000,2000,C).astype(np.int32)
            ref=dw_fast(act,w,m0,shift,bias,relu=1,pad=1,stride=st)
            r=dw_fpga(hw,act,w,m0,shift,bias,relu=1,pad=1,stride=st)
            d=np.abs(r.astype(np.int16)-ref.astype(np.int16))
            print(f"[DW-FPGA] C={C:3d} {H}x{W} s{st} out{r.shape}: within1={100*np.mean(d<=1):5.1f}% exact={100*np.mean(d==0):5.1f}% max={d.max():3d}")
        sys.exit(0)
    if "--c3onchiptest" in sys.argv:   # on-chip conv3x3 im2col (LAYER_TYPE=5) vs rr_unit-exact ref
        # last 3 are H1 ROW-BAND cases (Cin*WBin>16384 -> used to FALLBACK, now band-tiled on-chip)
        for (Cin,Cout,H,W) in [(4,16,6,6),(8,32,13,13),(16,32,26,26),(32,64,8,8),(64,64,13,13),(8,64,52,52),
                               (128,32,52,52),(64,32,104,104),(32,48,104,104)]:
            rng=np.random.default_rng(Cin*7+Cout)
            act=rng.integers(-60,60,(Cin,H,W)).astype(np.int8)
            w=rng.integers(-20,20,(Cout,Cin,3,3)).astype(np.int8)
            m0=rng.integers(0x3000,0x8000,Cout).astype(np.uint16); shift=rng.integers(16,20,Cout).astype(np.uint8)
            bias=rng.integers(-1500,1500,Cout).astype(np.int32)
            ref=ref_conv3x3_rr(act,w,m0,shift,bias,relu=1)
            r=conv3x3_onchip(hw,act,w,m0,shift,bias,relu=1)
            if r is None: print(f"[C3-ONCHIP] Cin={Cin} Cout={Cout} {H}x{W}: FALLBACK"); continue
            d=np.abs(r.astype(np.int16)-ref.astype(np.int16))
            print(f"[C3-ONCHIP] Cin={Cin:3d} Cout={Cout:3d} {H}x{W} out{r.shape}: within1={100*np.mean(d<=1):5.1f}% exact={100*np.mean(d==0):5.1f}% max={d.max():3d}")
        sys.exit(0)
    if "--sys3test" in sys.argv:
        # conv3x3 via im2col -> systolic, vs ref_conv3x3 (incl stride). within1 should be 100%.
        for (Cin,Cout,H,W,st) in [(8,12,7,7,1),(16,8,13,13,1),(32,16,26,26,1),(64,32,13,13,1),(32,16,16,16,2)]:
            rng=np.random.default_rng(Cin*5+Cout)
            act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8); w=rng.integers(-2,3,(Cout,Cin,3,3)).astype(np.int8)
            m0=rng.integers(0x2000,0x7000,Cout).astype(np.uint16); shift=rng.integers(14,18,Cout).astype(np.uint8); bias=rng.integers(-40,40,Cout).astype(np.int32)
            ref=ref_conv3x3(act,w,m0,shift,bias,relu=1,stride=st,pad=1)
            r=conv3x3_systolic(hw,act,w,m0,shift,bias,relu=1,pad=1,stride=st)
            if r is None: print(f"[SYS3] Cin={Cin} Cout={Cout} {H}x{W} s{st}: FALLBACK (wmem too big)"); continue
            d=np.abs(r.astype(np.int16)-ref.astype(np.int16))
            print(f"[SYS3 3x3] Cin={Cin:3d} Cout={Cout:2d} {H}x{W} s{st} out{r.shape}: within1={100*np.mean(d<=1):5.1f}% max={d.max():3d}")
        sys.exit(0)
    if "--systolicbig" in sys.argv:
        # REAL-scale conv1x1 layers (depth 4096 + OUT-chunk). Correctness + HW-kick timing vs kernel cũ.
        import time
        def kick_us(n=9):
            xs=[]
            for _ in range(n): t=time.time(); hw.kick(); xs.append(time.time()-t)
            return np.median(xs)*1e6
        for (Cin,Cout,H,W) in [(256,128,13,13),(256,256,13,13),(64,64,26,26)]:
            rng=np.random.default_rng(Cin+Cout)
            act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8); w1=rng.integers(-2,3,(Cout,Cin,1,1)).astype(np.int8)
            m0=rng.integers(0x2000,0x7000,Cout).astype(np.uint16); shift=rng.integers(14,18,Cout).astype(np.uint8); bias=rng.integers(-40,40,Cout).astype(np.int32)
            w3=np.zeros((Cout,Cin,3,3),np.int8); w3[:,:,1,1]=w1[:,:,0,0]
            ref=ref_conv3x3(act,w3,m0,shift,bias,relu=1,pad=1)
            r=conv1x1_systolic(hw,act,w1,m0,shift,bias,relu=1)   # correctness (also leaves layout in DDR)
            d=np.abs(r.astype(np.int16)-ref.astype(np.int16))
            # re-arm systolic regs and time HW-kick only (data already in DDR from the call above)
            HWn=H*W; WB=(HWn+16-1)//16; N_OCB=(Cout+15)//16; nwmem=N_OCB*Cin; npar=(Cout+1)//2
            hw.wr(REG_LAYER_TYPE,3); hw.wr(REG_CIN,Cin); hw.wr(REG_COUT,Cout); hw.wr(REG_HIN,HWn); hw.wr(REG_WIN,1)
            hw.wr64(REG_IN_LO,hw.phys+B0_ACT); hw.wr64(REG_WT_LO,hw.phys+B0_WT); hw.wr64(REG_OUT_LO,hw.phys+B0_OUT)
            hw.wr(REG_WT_BYTES,(nwmem+npar)*16); hw.wr(REG_OUT_BYTES,Cout*WB*16)
            sk=kick_us(); hw.wr(REG_LAYER_TYPE,0)
            tb=time.time(); _=conv1x1_layer_B0(hw,act,w1,m0,shift,bias,relu=1); bms=(time.time()-tb)*1e3
            mac=Cin*Cout*HWn
            print(f"[BIG 1x1] Cin={Cin} Cout={Cout} {H}x{W} ({mac/1e6:.1f}MMAC): within1={100*np.mean(d<=1):5.1f}% | systolic HW-kick={sk:.0f}us  B0_old_fullcall={bms:.1f}ms  speedup~{bms*1000/sk:.0f}x")
        sys.exit(0)
    if "--selftest" in sys.argv:
        # single tile: Cin channels, 4 out-ch, one output row (Wout=16). Random small.
        rng=np.random.default_rng(2)
        _cin=[a.split('=')[1] for a in sys.argv if a.startswith('--cin=')]
        Cin = int(_cin[0]) if _cin else 4
        act = rng.integers(-15,15,(Cin,KH,KW)).astype(np.int8)
        w   = rng.integers(-2,3,(4,Cin,3,3)).astype(np.int8)
        m0  = np.full(4,0x4000,np.uint16); shift=np.full(4,15,np.uint8); bias=np.zeros(4,np.int32)
        out = conv3x3_tile(hw, act, w, m0, shift, bias, relu=1)
        # RTL does VALID conv on the 3x18 tile (host supplies halo) → 1x16. ref pad=0.
        ref = ref_conv3x3(act, w, m0, shift, bias, relu=1, pad=0)[:, 0, :WOUT]
        d=np.abs(out.astype(np.int16)-ref.astype(np.int16))
        print(f"[conv3x3 selftest] Cin={Cin}")
        for g in range(4):
            print(f"  oc{g} FPGA={out[g].tolist()}")
            print(f"  oc{g} ref ={ref[g].tolist()}")
            bad=np.where(d[g]>1)[0]
            if len(bad): print(f"      diff>1 at p={bad.tolist()}  d={d[g][bad].tolist()}")
        print(f"  within1={100*np.mean(d<=1):.1f}% max={d.max()}")
        print("PASS" if d.max()<=1 else "FAIL")
        sys.exit(0)
    if "--layertest" in sys.argv:
        rng=np.random.default_rng(3)
        Cin,Cout,H,W = 4, 6, 5, 20    # 2 out-blocks (4+2), multi-row, 2 col-blocks
        act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8)
        w=rng.integers(-2,3,(Cout,Cin,3,3)).astype(np.int8)
        m0=np.full(Cout,0x4000,np.uint16); shift=np.full(Cout,15,np.uint8); bias=np.zeros(Cout,np.int32)
        ref=ref_conv3x3(act,w,m0,shift,bias,relu=1,pad=1)
        out=conv3x3_layer(hw,act,w,m0,shift,bias,relu=1,pad=1)
        d=np.abs(out.astype(np.int16)-ref.astype(np.int16))
        print(f"[conv3x3 LAYER] Cin={Cin} Cout={Cout} {H}x{W}")
        print(f"  within1={100*np.mean(d<=1):.1f}% max={d.max()}")
        print("PASS" if d.max()<=1 else "FAIL")
        sys.exit(0)
    if "--b0test" in sys.argv:
        # B0 on-chip layer_sequencer vs ref + vs per-tile conv3x3_layer (bit-exact)
        for (Cin,Cout,H,W,st) in [(4,6,5,20,1),(32,8,6,20,1),(64,12,7,18,1),(4,6,8,20,2),
                                  (256,8,13,13,1),(512,12,13,13,1)]:   # N_ICB=16,32 (5-bit overflow guard)
            rng=np.random.default_rng(Cin+st)
            act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8)
            w=rng.integers(-2,3,(Cout,Cin,3,3)).astype(np.int8)
            m0=rng.integers(0x2000,0x7000,Cout).astype(np.uint16); shift=rng.integers(14,18,Cout).astype(np.uint8)
            bias=rng.integers(-40,40,Cout).astype(np.int32)
            ref=ref_conv3x3(act,w,m0,shift,bias,relu=1,pad=1,stride=st)
            old=conv3x3_layer(hw,act,w,m0,shift,bias,relu=1,pad=1,stride=st)
            b0 =conv_layer_B0(hw,act,w,m0,shift,bias,relu=1,pad=1,stride=st)
            dr=np.abs(b0.astype(np.int16)-ref.astype(np.int16))
            print(f"[B0] Cin={Cin:3d} Cout={Cout:2d} {H}x{W} s{st} out{b0.shape}: "
                  f"vs ref within1={100*np.mean(dr<=1):5.1f}% max={dr.max():3d} | "
                  f"B0==per-tile {np.array_equal(b0,old)}")
        sys.exit(0)
    if "--r3test" in sys.argv:
        # R3 conv1x1 (1-tap kernel) vs ref (center-tap 3x3 = 1x1) — multi Cin incl 512 (32 chunk)
        for (Cin,Cout,H,W) in [(16,4,4,16),(32,8,6,20),(64,12,13,13),(256,8,13,13),(512,12,13,13)]:
            rng=np.random.default_rng(Cin*3+Cout)
            act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8)
            w1=rng.integers(-2,3,(Cout,Cin,1,1)).astype(np.int8)
            m0=rng.integers(0x2000,0x7000,Cout).astype(np.uint16); shift=rng.integers(14,18,Cout).astype(np.uint8)
            bias=rng.integers(-40,40,Cout).astype(np.int32)
            w3=np.zeros((Cout,Cin,3,3),np.int8); w3[:,:,1,1]=w1[:,:,0,0]   # center-tap 3x3 = 1x1
            ref=ref_conv3x3(act,w3,m0,shift,bias,relu=1,pad=1)
            r3 =conv1x1_layer_B0(hw,act,w1,m0,shift,bias,relu=1)
            d=np.abs(r3.astype(np.int16)-ref.astype(np.int16))
            print(f"[R3 1x1] Cin={Cin:3d} Cout={Cout:2d} {H}x{W} out{r3.shape}: vs ref within1={100*np.mean(d<=1):5.1f}% max={d.max():3d}")
        sys.exit(0)
    if "--stride2test" in sys.argv:
        rng=np.random.default_rng(7)
        Cin,Cout,H,W = 4, 6, 8, 20    # stride2, 2 out-blocks, multi-row, 2 col-blocks
        act=rng.integers(-15,15,(Cin,H,W)).astype(np.int8)
        w=rng.integers(-3,3,(Cout,Cin,3,3)).astype(np.int8)
        m0=np.full(Cout,0x4000,np.uint16); shift=np.full(Cout,15,np.uint8); bias=rng.integers(-50,50,Cout).astype(np.int32)
        ref=ref_conv3x3(act,w,m0,shift,bias,relu=1,pad=1,stride=2)
        out=conv3x3_layer(hw,act,w,m0,shift,bias,relu=1,pad=1,stride=2)
        d=np.abs(out.astype(np.int16)-ref.astype(np.int16))
        print(f"[conv3x3 STRIDE2] Cin={Cin} Cout={Cout} {H}x{W} -> out{out.shape} ref{ref.shape}")
        # sanity: native stride2 must equal stride1-then-[::2,::2] (the thing #1 replaces)
        s1=conv3x3_layer(hw,act,w,m0,shift,bias,relu=1,pad=1,stride=1)[:,::2,::2]
        print(f"  vs ref:    within1={100*np.mean(d<=1):.1f}% max={d.max()}")
        print(f"  vs stride1[::2]: equal={np.array_equal(out,s1)} (must be True)")
        print("PASS" if d.max()<=1 and np.array_equal(out,s1) else "FAIL")
        sys.exit(0)
    if "--dwtest" in sys.argv:
        rng=np.random.default_rng(4)
        nch=4
        act=rng.integers(-15,15,(nch,KH,KW)).astype(np.int8)   # 4 channels
        w=rng.integers(-2,3,(nch,3,3)).astype(np.int8)
        m0=np.full(nch,0x4000,np.uint16); sh=np.full(nch,15,np.uint8); bias=np.zeros(nch,np.int32)
        out=dwconv_tile(hw,act,w,m0,sh,bias,relu=1)            # [4,16]
        # ref: per-channel valid dw conv on 3x18 → 1x16 (depthwise, w[c] only on act[c])
        ref=np.zeros((nch,WOUT),np.int8)
        for c in range(nch):
            ref[c]=ref_conv3x3(act[c:c+1],w[c][None,None],np.array([m0[c]],np.uint16),np.array([sh[c]],np.uint8),np.array([bias[c]],np.int32),1,pad=0)[0,0,:WOUT]
        d=np.abs(out.astype(np.int16)-ref.astype(np.int16))
        for c in range(nch):
            print(f"  ch{c} FPGA={out[c,:8].tolist()} ref={ref[c,:8].tolist()}")
        print(f"[dwconv 4-ch] within1={100*np.mean(d<=1):.1f}% max={d.max()}")
        print("PASS" if d.max()<=1 else "FAIL")
        sys.exit(0)
    if "--dwlayertest" in sys.argv:
        rng=np.random.default_rng(5)
        C,H,W=6,5,20
        act=rng.integers(-15,15,(C,H,W)).astype(np.int8)
        w=rng.integers(-2,3,(C,3,3)).astype(np.int8)
        m0=np.full(C,0x4000,np.uint16); sh=np.full(C,15,np.uint8); bias=np.zeros(C,np.int32)
        out=dwconv_layer(hw,act,w,m0,sh,bias,relu=1,pad=1)
        ref=np.zeros((C,H,W),np.int8)
        for c in range(C):
            ref[c]=ref_conv3x3(act[c:c+1],w[c][None,None],np.array([m0[c]],np.uint16),np.array([sh[c]],np.uint8),np.array([bias[c]],np.int32),1,pad=1)[0]
        d=np.abs(out.astype(np.int16)-ref.astype(np.int16))
        print(f"[dwconv LAYER] C={C} {H}x{W} within1={100*np.mean(d<=1):.1f}% max={d.max()}")
        print("PASS" if d.max()<=1 else "FAIL")
        sys.exit(0)
    if "--dwb0test" in sys.argv:
        rng=np.random.default_rng(7)
        for (C,H,W) in [(6,5,20),(8,6,20),(32,13,13),(64,26,26),(128,52,52)]:
            act=rng.integers(-15,15,(C,H,W)).astype(np.int8)
            w=rng.integers(-2,3,(C,3,3)).astype(np.int8)
            m0=rng.integers(0x2000,0x7000,C).astype(np.uint16); sh=rng.integers(14,18,C).astype(np.uint8); bias=rng.integers(-40,40,C).astype(np.int32)
            out=dwconv_layer_B0(hw,act,w,m0,sh,bias,relu=1,pad=1)
            ref=np.zeros((C,H,W),np.int8)
            for c in range(C):
                ref[c]=ref_conv3x3(act[c:c+1],w[c][None,None],np.array([m0[c]],np.uint16),np.array([sh[c]],np.uint8),np.array([bias[c]],np.int32),1,pad=1)[0]
            d=np.abs(out.astype(np.int16)-ref.astype(np.int16))
            print(f"[dwB0] C={C:3d} {H}x{W} out{out.shape}: within1={100*np.mean(d<=1):.1f}% max={d.max()}")
        sys.exit(0)
    print("usage: fpga_conv3x3.py --selftest [--cin=N] | --layertest | --dwtest | --dwlayertest | --dwb0test")

def conv1x1_as_conv3x3(hw, act, w1x1, m0, shift, bias, relu=1):
    """Dense conv1x1 via conv3x3 center-tap (conv1x1 IP can't do dense N:M). w1x1[Cout,Cin,1,1]."""
    Cout, Cin = w1x1.shape[0], w1x1.shape[1]
    w3 = np.zeros((Cout, Cin, 3, 3), np.int8)
    w3[:, :, 1, 1] = w1x1.reshape(Cout, Cin)
    return conv3x3_layer(hw, act, w3, m0, shift, bias, relu=relu, pad=1)


# ===== B1 on-chip layer-sequencer (appended) =====
def conv1x1_systolic_seq(hw, act, w1, m0, shift, bias, relu=1, to=60.0):
    import numpy as np, time
    N=16; _worig=id(w1)
    w1=w1.reshape(w1.shape[0], w1.shape[1])
    Cout,Cin=w1.shape; _,H,W=act.shape; HWn=H*W
    # tile dims (uniform). cchunk = Cout/tile (wmem-bound), cmax = HW/tile (amem+omem-bound).
    # tile dims: distribute Cout/HW EVENLY across the min # of tiles (minimal padding).
    ocb = max(1, (SYS_ADEPTH-512)//Cin)
    max_cchunk = min(ocb*N, ((Cout+N-1)//N)*N)
    n_cot = (Cout + max_cchunk - 1)//max_cchunk
    cchunk = (((Cout + n_cot - 1)//n_cot) + N-1)//N*N      # even split, round up to 16
    max_cmax = min(1008, (SYS_ODEPTH*N)//cchunk, (SYS_ADEPTH*N)//Cin)  # 1008<=1023 (reg_hin 10-bit)
    max_cmax = max(N, (max_cmax//N)*N)
    n_hwt = (HWn + max_cmax - 1)//max_cmax
    cmax = (((HWn + n_hwt - 1)//n_hwt) + N-1)//N*N         # even split, round up to 16
    HWp = n_hwt*cmax; Coutp = n_cot*cchunk
    WB = cmax//N
    # config per (uniform) tile
    nwmem = (cchunk//N)*Cin; npar = (cchunk+1)//2
    act_tile_bytes = Cin*WB*N            # banked act blob per HW-tile
    wt_tile_bytes  = (nwmem+npar)*N      # [wmem][pmem] blob per Cout-tile
    out_tile_bytes = cchunk*WB*N         # omem out per (cot,hwt) tile

    # ── ACT layout: bank each hwt-tile (cmax positions) -> DDR @ B0_ACT + hwt*act_tile_bytes ──
    af = np.ascontiguousarray(act.reshape(Cin, HWn))
    if HWp != HWn:
        afp = np.zeros((Cin, HWp), np.int8); afp[:, :HWn] = af
    else:
        afp = af
    # amem[ci*WB + p][lane] = A[ci][p*16+lane]; per tile it's just the contiguous cmax slice (HW mult 16)
    act_blob = np.empty((n_hwt, Cin*WB*N), np.int8)
    for t in range(n_hwt):
        act_blob[t] = afp[:, t*cmax:(t+1)*cmax].reshape(Cin, WB*N).reshape(-1)
    hw.put(B0_ACT, act_blob.reshape(-1))

    # ── WT layout: [wmem][pmem] blob per cot-tile -> DDR @ B0_WT + cot*wt_tile_bytes (cached) ──
    _wk=(_worig,'seq',Cout,Cin,cchunk)
    wt=_WT_CACHE.get(_wk)
    if wt is None:
        wp=np.zeros((Coutp, Cin), np.int8); wp[:Cout]=w1.astype(np.int8)
        m0p=np.zeros(Coutp,np.uint16);   m0p[:Cout]=np.asarray(m0,np.uint16)
        shp=np.zeros(Coutp,np.uint8);    shp[:Cout]=np.asarray(shift,np.uint8)
        bip=np.zeros(Coutp,np.int32);    bip[:Cout]=np.asarray(bias,np.int32)
        blobs=[]
        for c in range(n_cot):
            cw=wp[c*cchunk:(c+1)*cchunk]                    # [cchunk,Cin]
            wmem=np.ascontiguousarray(cw.reshape(cchunk//N,N,Cin).transpose(0,2,1)).reshape(nwmem,N)
            half=((bip[c*cchunk:(c+1)*cchunk].astype(np.int64)&0xFFFFFFFF).astype(np.uint64)<<24)\
                 |((shp[c*cchunk:(c+1)*cchunk].astype(np.uint64)&0x1F)<<16)\
                 |(m0p[c*cchunk:(c+1)*cchunk].astype(np.uint64)&0xFFFF)
            hpad=np.zeros(npar*2,np.uint64); hpad[:cchunk]=half
            pmem=hpad.view(np.uint8).reshape(npar,N)
            blobs.append(np.concatenate([wmem.reshape(-1).view(np.int8), pmem.reshape(-1).view(np.int8)]))
        wt=np.concatenate(blobs)
        _WT_CACHE[_wk]=wt
    hw.put(B0_WT, wt)

    # ── config regs: engine sees TILE dims (uniform); sequencer sees n_hwt/n_cot + strides ──
    hw.wr(REG_APPLY_RELU, 1 if relu else 0)
    hw.wr(REG_LAYER_TYPE, 3)                                 # conv1x1 systolic
    hw.wr(REG_CIN, Cin); hw.wr(REG_COUT, cchunk); hw.wr(REG_HIN, cmax); hw.wr(REG_WIN, 1)
    hw.wr64(REG_IN_LO, hw.phys+B0_ACT); hw.wr64(REG_WT_LO, hw.phys+B0_WT); hw.wr64(REG_OUT_LO, hw.phys+B0_OUT)
    hw.wr(REG_WT_BYTES, wt_tile_bytes); hw.wr(REG_OUT_BYTES, out_tile_bytes)
    hw.wr(REG_N_POS, n_hwt); hw.wr(REG_N_OCB, n_cot)
    hw.wr(REG_ACT_CHUNK, act_tile_bytes); hw.wr(REG_WT_BLK_BYTES, wt_tile_bytes)
    hw.wr(REG_LAYER_MODE, 1)
    _kick(hw, 'c1seq', to)
    hw.wr(REG_LAYER_MODE, 0); hw.wr(REG_LAYER_TYPE, 0)

    # ── get all OUT tiles + de-interleave: tile(cot,hwt) @ idx (cot*n_hwt+hwt) ──
    total = n_cot*n_hwt*out_tile_bytes
    raw = hw.get(B0_OUT, total).reshape(n_cot, n_hwt, cchunk, WB, N).reshape(n_cot, n_hwt, cchunk, WB*N)
    out = np.zeros((Coutp, HWp), np.int8)
    for c in range(n_cot):
        for t in range(n_hwt):
            out[c*cchunk:(c+1)*cchunk, t*cmax:(t+1)*cmax] = raw[c,t]
    return out[:Cout, :HWn].reshape(Cout, H, W)
