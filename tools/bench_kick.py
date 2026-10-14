#!/usr/bin/env python3
"""bench_kick.py — ground-truth per-phase timing of ONE conv3x3 kick on the board.
Splits the 0.224ms/kick into: put_act (DMA), reg writes, ap_idle wait, IP compute
(ap_start->ap_done), dbg drain, get_out (readback). Tells us exactly where the host
bottleneck is and what is/ isn't optimizable. Run: sudo python3 bench_kick.py"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/ubuntu/rtl_bd')
import fpga_conv3x3 as F
from fpga_conv3x3 import HW

hw = HW()
Cin = 16
rng = np.random.default_rng(1)
w  = rng.integers(-2,3,(4,Cin,3,3)).astype(np.int8)
m0 = np.full(4,0x4000,np.uint16); sh=np.full(4,15,np.uint8); b=np.zeros(4,np.int32)
blob = np.frombuffer(F.encode_block(w,m0,sh,b), np.int8)
act  = rng.integers(-15,15,(Cin,F.KH,F.KW)).astype(np.int8)
aflat= np.ascontiguousarray(act.reshape(-1), np.int8)
arow = np.ascontiguousarray(rng.integers(-15,15,(Cin,F.KH,F.KW)).reshape(-1).astype(np.int8))

AP, DBG = F.REG_AP_CTRL, F.REG_DBG_STATE
# preload WT + constant regs (as conv3x3_layer does once/block)
hw.put(F.OFF_WT, blob)
for r,v in [(F.REG_LAYER_TYPE,1),(F.REG_COUT,4),(F.REG_HIN,F.KH),(F.REG_WIN,F.KW),
            (F.REG_N_CONFIG,1),(F.REG_POC,2),(F.REG_ACCU_MODE,0),(F.REG_APPLY_RELU,1),
            (F.REG_STRIDE,1),(F.REG_OUT_BYTES,64),(F.REG_WT_BYTES,len(blob)),
            (F.REG_CIN,Cin),(F.REG_PAD,F._pad_ctrl(False,False))]:
    hw.wr(r,v)
hw.wr64(F.REG_IN_LO,hw.phys+F.OFF_ACT); hw.wr64(F.REG_WT_LO,hw.phys+F.OFF_WT); hw.wr64(F.REG_OUT_LO,hw.phys+F.OFF_OUT)
hw.put(F.OFF_ACT,aflat); hw.kick()   # warm prime

N=400
tp=tr=ti=tc=td=tg=0.0
for i in range(N):
    t0=time.perf_counter(); hw.put(F.OFF_ACT,aflat); t1=time.perf_counter()
    hw.wr(F.REG_CIN,Cin); hw.wr(F.REG_WT_BYTES,len(blob)); hw.wr(F.REG_PAD,F._pad_ctrl(False,False)); hw.wr64(F.REG_WT_LO,hw.phys+F.OFF_WT)
    t2=time.perf_counter()
    while not (hw.rd(AP)&F.AP_IDLE): pass
    t3=time.perf_counter(); hw.wr(AP,F.AP_START)
    while not (hw.rd(AP)&F.AP_DONE): pass
    t4=time.perf_counter()
    while (hw.rd(DBG)&0xF)!=0: pass
    t5=time.perf_counter(); _=hw.get(F.OFF_OUT,64); t6=time.perf_counter()
    tp+=t1-t0; tr+=t2-t1; ti+=t3-t2; tc+=t4-t3; td+=t5-t4; tg+=t6-t5

u=lambda s: s/N*1e6
tot=u(tp)+u(tr)+u(ti)+u(tc)+u(td)+u(tg)
print(f"  put_act (DMA memmove)   {u(tp):7.1f} us")
print(f"  reg_wr x5 (MMIO)        {u(tr):7.1f} us")
print(f"  ap_idle wait            {u(ti):7.1f} us")
print(f"  IP compute (start->done){u(tc):7.1f} us")
print(f"  dbg drain               {u(td):7.1f} us")
print(f"  get_out (readback)      {u(tg):7.1f} us")
print(f"  --------------------------------")
print(f"  TOTAL / kick            {tot:7.1f} us")
# raw MMIO read + write cost
t0=time.perf_counter()
for _ in range(20000): hw.rd(AP)
rdus=(time.perf_counter()-t0)/20000*1e6
t0=time.perf_counter()
for _ in range(20000): hw.wr(F.REG_CIN,Cin)
wrus=(time.perf_counter()-t0)/20000*1e6
print(f"  [1 MMIO read={rdus:.3f}us  1 MMIO write={wrus:.3f}us]")
# how many MMIO reads does the compute+drain poll burn?
print(f"  [compute+drain poll ~= {(u(tc)+u(td))/rdus:.0f} MMIO reads @ {rdus:.3f}us]")
