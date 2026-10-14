#!/usr/bin/env python3
"""bench_dw.py — so 3 impl DW-CPU trên shape DW thật, verify bit-exact + đo ms."""
import json, os, time, numpy as np

def rq(acc, bias, M0, sh, relu):
    acc = acc.astype(np.int64) + bias.astype(np.int64)[:, None, None]
    prod = acc * M0.astype(np.int64)[:, None, None]
    s64 = sh.astype(np.int64)[:, None, None]
    sc = np.where(s64 > 0, (prod + (np.int64(1) << np.maximum(s64-1, 0))) >> s64, prod)
    if relu: sc = np.where(sc >= 0, sc, (sc*np.int64(205)+np.int64(1024)) >> 11)
    return np.clip(sc, -128, 127).astype(np.int8)

def v0(x, w, bias, M0, sh, relu):          # bản hiện tại: pad int32, 9 pass toàn tensor
    C,H,W = x.shape
    xp = np.pad(x.astype(np.int32), ((0,0),(1,1),(1,1)))
    acc = np.zeros((C,H,W), np.int32)
    for r in range(3):
        for c in range(3):
            acc += xp[:, r:r+H, c:c+W] * w[:,0,r,c].astype(np.int32)[:,None,None]
    return rq(acc, bias, M0, sh, relu)

def v1(x, w, bias, M0, sh, relu):          # pad int8, product int16, 9 pass
    C,H,W = x.shape
    xp = np.pad(x, ((0,0),(1,1),(1,1)))    # int8!
    acc = np.zeros((C,H,W), np.int32)
    w16 = w[:,0].astype(np.int16)
    for r in range(3):
        for c in range(3):
            np.add(acc, xp[:, r:r+H, c:c+W] * w16[:,r,c][:,None,None], out=acc, casting='unsafe')
    return rq(acc, bias, M0, sh, relu)

def v2(x, w, bias, M0, sh, relu, BLK=8):   # v1 + block kênh (cache-hot)
    C,H,W = x.shape
    out = np.empty((C,H,W), np.int8)
    w16 = w[:,0].astype(np.int16)
    for c0 in range(0, C, BLK):
        c1 = min(c0+BLK, C)
        xp = np.pad(x[c0:c1], ((0,0),(1,1),(1,1)))
        acc = np.zeros((c1-c0,H,W), np.int32)
        for r in range(3):
            for c in range(3):
                np.add(acc, xp[:, r:r+H, c:c+W] * w16[c0:c1,r,c][:,None,None], out=acc, casting='unsafe')
        out[c0:c1] = rq(acc, bias[c0:c1], M0[c0:c1], sh[c0:c1], relu)
    return out

shapes = json.load(open(os.path.expanduser('~/golden/layer_shapes.json')))
DW = [7,14,21,28,37,44,51,58]
rng = np.random.default_rng(1)
tot = {'v0':0,'v1':0,'v2':0}
for lid in DW:
    C,H,W = shapes[f'layer_{lid:02d}']['in']
    x = rng.integers(-128,128,(C,H,W)).astype(np.int8)
    w = rng.integers(-128,128,(C,1,3,3)).astype(np.int8)
    bias = rng.integers(-20000,20000,C).astype(np.int32)
    M0 = rng.integers(8000,65536,C).astype(np.uint16)
    sh = rng.integers(16,25,C).astype(np.uint8)
    r0 = v0(x,w,bias,M0,sh,True)
    for name, fn in [('v1',v1),('v2',v2)]:
        ri = fn(x,w,bias,M0,sh,True)
        assert (ri==r0).all(), f'{name} MISMATCH L{lid}'
    for name, fn in [('v0',v0),('v1',v1),('v2',v2)]:
        t=time.time(); fn(x,w,bias,M0,sh,True); dt=(time.time()-t)*1e3
        tot[name]+=dt
    print(f'L{lid:02d} {C}x{H}x{W}')
print('TONG (ms):', {k: round(v,1) for k,v in tot.items()}, ' BIT-EXACT v1=v2=v0 OK')

def v3(x, w, bias, M0, sh, relu):          # v1 + tmp buffer tái dùng (không alloc/tap)
    C,H,W = x.shape
    xp = np.pad(x, ((0,0),(1,1),(1,1)))
    acc = np.zeros((C,H,W), np.int32)
    tmp = np.empty((C,H,W), np.int16)
    w16 = w[:,0].astype(np.int16)
    for r in range(3):
        for c in range(3):
            np.multiply(xp[:, r:r+H, c:c+W], w16[:,r,c][:,None,None], out=tmp, casting='unsafe')
            np.add(acc, tmp, out=acc, casting='unsafe')
    return rq(acc, bias, M0, sh, relu)

def v2b(x,w,bias,M0,sh,relu): return v2(x,w,bias,M0,sh,relu,BLK=32)

def v5(x, w, bias, M0, sh, relu):          # sliding_window_view + einsum
    from numpy.lib.stride_tricks import sliding_window_view
    C,H,W = x.shape
    xp = np.pad(x, ((0,0),(1,1),(1,1)))
    win = sliding_window_view(xp, (3,3), axis=(1,2))          # (C,H,W,3,3) view int8
    acc = np.einsum('chwij,cij->chw', win.astype(np.int16), w[:,0].astype(np.int16)).astype(np.int32)
    return rq(acc, bias, M0, sh, relu)

tot2 = {'v1':0,'v3':0,'v2b':0,'v5':0}
for lid in DW:
    C,H,W = shapes[f'layer_{lid:02d}']['in']
    x = rng.integers(-128,128,(C,H,W)).astype(np.int8)
    w = rng.integers(-128,128,(C,1,3,3)).astype(np.int8)
    bias = rng.integers(-20000,20000,C).astype(np.int32)
    M0 = rng.integers(8000,65536,C).astype(np.uint16)
    sh = rng.integers(16,25,C).astype(np.uint8)
    r0 = v1(x,w,bias,M0,sh,True)
    for name, fn in [('v3',v3),('v2b',v2b),('v5',v5)]:
        assert (fn(x,w,bias,M0,sh,True)==r0).all(), f'{name} MISMATCH'
    for name, fn in [('v1',v1),('v3',v3),('v2b',v2b),('v5',v5)]:
        t=time.time(); fn(x,w,bias,M0,sh,True); tot2[name]+=(time.time()-t)*1e3
print('VONG 2 (ms):', {k: round(v,1) for k,v in tot2.items()})
