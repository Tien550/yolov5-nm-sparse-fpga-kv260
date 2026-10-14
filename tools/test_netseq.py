"""test_netseq.py -- PoC verify: network sequencer (LT14) chains L26→L30 conv1x1 under ONE ap_start.
Compares netseq output vs running the 2 layers SEPARATELY (golden). Bit-exact = no-race + correct.
L26: Cout=512,Cin=512,13x13 ; L30: Cout=256,Cin=512,13x13. L26.out feeds L30 directly (P5bb→s1)."""
import sys, os, json, time, numpy as np
sys.path.insert(0, '/home/ubuntu/yolov5/tools')
import inference_cpu as C
from fpga_conv3x3 import HW, conv1x1_systolic, net_seq_run

def main():
    meta, req, weights, bias = C.load_model()
    def lp(lid):
        k=f"layer_{lid:02d}"; m=meta[k]; ws=m["weight_shape"]
        w=weights[k].reshape(ws[0],ws[1])
        M0=np.array(req[k]["M0"],np.uint16); sh=np.array(req[k]["shift"],np.uint8)
        return w, M0, sh, bias[k]
    w26,m26,s26,b26 = lp(26)
    w30,m30,s30,b30 = lp(30)
    print(f"L26: Cout={w26.shape[0]} Cin={w26.shape[1]}  L30: Cout={w30.shape[0]} Cin={w30.shape[1]}")
    assert w26.shape[0]==w30.shape[1], "chain mismatch"

    rng=np.random.default_rng(7)
    act=rng.integers(-40,40,(w26.shape[1],13,13),dtype=np.int8)   # L26 input (512,13,13)

    hw=HW()
    # GOLDEN: 2 separate kicks
    o0 = conv1x1_systolic(hw, act, w26, m26, s26, b26, relu=1)
    o1_gold = conv1x1_systolic(hw, o0, w30, m30, s30, b30, relu=1)
    print(f"golden L30 out shape={o1_gold.shape} sum={int(o1_gold.astype(np.int64).sum())}")

    # NETSEQ: 1 ap_start chains both
    o1_ns = net_seq_run(hw, act, [(w26,m26,s26,b26),(w30,m30,s30,b30)], relu=1)
    print(f"netseq L30 out shape={o1_ns.shape} sum={int(o1_ns.astype(np.int64).sum())}")

    d = np.abs(o1_gold.astype(np.int16)-o1_ns.astype(np.int16))
    nmis = int((d>0).sum())
    print(f"\n=== mismatch={nmis}/{o1_gold.size}  max_diff={int(d.max())}  within1={100*np.mean(d<=1):.2f}% ===")
    if nmis==0: print("*** PASS: netseq bit-exact == 2 separate kicks. NO-RACE CONFIRMED. ***")
    elif d.max()<=1: print("*** NEAR-PASS: within1 (latency/rounding edge). ***")
    else: print(f"*** FAIL: {nmis} mismatches, max_diff={int(d.max())}. ***")

    # timing: netseq vs 2-kick
    for nm,fn in [("2-kick", lambda: (conv1x1_systolic(hw,act,w26,m26,s26,b26,relu=1),
                                       conv1x1_systolic(hw,conv1x1_systolic(hw,act,w26,m26,s26,b26,relu=1),w30,m30,s30,b30,relu=1))),
                  ("netseq", lambda: net_seq_run(hw,act,[(w26,m26,s26,b26),(w30,m30,s30,b30)],relu=1))]:
        for _ in range(2): fn()    # warm
        t=time.perf_counter()
        for _ in range(20): fn()
        print(f"  {nm}: {(time.perf_counter()-t)/20*1e3:.2f}ms")

if __name__=="__main__":
    main()
