"""
inference_cpu.py -- All-CPU YOLOv5s-Lite inference (numpy) using cpu_conv.

A correct, verifiable reference inference: every conv (1x1/3x3/dw) runs on CPU via
cpu_conv.quant_conv (validated within BN-rounding noise against per-layer goldens).
Graph ported from drivers/kv260_test/inference_unified.py::run_graph.

Purpose: a working end-to-end inference (backbone + PAN + 3-scale detect head + NMS)
independent of the FPGA, to (a) validate the model/pipeline end-to-end, and (b) serve
as ground truth when bringing up conv3x3/dwconv on FPGA. Slow (pure numpy conv) but
correct. Run with --verify to check each layer against golden/L{nn}_out_CHW.bin.

Usage:
  python tools/inference_cpu.py --verify          # run from golden input, check layers
  python tools/inference_cpu.py --image x.jpg     # run on a real image
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cpu_conv import quant_conv, float_conv, quantize_to_int8

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.environ.get("SPARSE_EXTRACTED",
    os.path.join(ROOT, "yolov5/runs/quant/extracted"))
GOLD = os.path.join(ROOT, "golden")

IMG_SIZE = int(os.environ.get('IMGSZ', '416'))   # env IMGSZ=256 để chạy resolution thấp (realtime lever)
NUM_CLASSES = 2
CLASS_NAMES = ['space-empty', 'space-occupied']
INPUT_SCALE = 0.007885098084807396
ANCHORS = [[[11,24],[13,30],[15,26]], [[17,34],[22,35],[30,27]], [[27,44],[37,38],[48,63]]]
STRIDES = [8, 16, 32]
CONF_THRESH, NMS_THRESH = float(os.environ.get('CONF','0.25')), float(os.environ.get('NMS','0.12'))   # env-overridable for tuning. CLASS-AWARE NMS @ iou=0.12. agnostic@0.20 left 69 stacked pairs;
                                        # agnostic@0.10 ate real adjacent slots (116->85). Class-aware
                                        # (empty/occupied NMS'd separately) keeps adjacent diff-class slots
                                        # while dropping same-class dups. 0.15 still left a few stacks ->
                                        # 0.12 tightens it: ~98 dets, cleaner, still keeps adjacent slots.
# AGN_IOU>0: sau class-aware, thêm 1 lượt agnostic (mọi lớp) xoá duplicate CHÉO lớp — 1 ô đỗ
# không đồng thời empty+occupied. IoU 0.20 giữ slot KỀ (khác slot), chỉ bỏ box mâu thuẫn cùng-ô.
# Default 0 = tắt (giữ golden 93 det). Deploy sạch: AGN_IOU=0.20 → ~77 det, hết chồng chéo.
AGN_IOU = float(os.environ.get('AGN_IOU', '0'))
MIN_BOX_PX, MAX_BOX_PX = 12, 80
MIN_ASPECT, MAX_ASPECT = 0.20, 5.0
CENTER_DIST_PX, MAX_DETS = 20, 300
FP32_NO_LEAKY = {42, 49, 56, 60, 61, 62}   # detect-head / fp32-out: no leaky relu


def load_model():
    meta = json.load(open(f"{E}/metadata.json"))
    req = json.load(open(f"{E}/requant_params.json"))
    weights, bias = {}, {}
    for lid in range(63):
        k = f"layer_{lid:02d}"
        m = meta[k]
        weights[k] = np.fromfile(f"{E}/weights_bin/{k}.bin", dtype=np.int8).reshape(m["weight_shape"])
        bias[k] = np.fromfile(f"{E}/bias_bin/{k}.bin", dtype=np.int32)
    return meta, req, weights, bias


# ── inter-layer helpers (ported from inference_unified.py) ───────────────────
# GPROF: env-gated per-op timing of the CPU inter-layer graph ops (to scope what
# CPU work is worth moving to FPGA). Accumulates wall time per bucket.
_GPROF = {}
_GPON = bool(os.environ.get('GPROF'))
def _gp(name, t0):
    if _GPON: _GPROF[name] = _GPROF.get(name, 0.0) + time.time() - t0

def rescale_int8(x, fs, ts):
    # NOTE: LUT-gather was tried but SLOWER (cat 29→37ms) — numpy SIMD float multiply beats fancy-index
    # gather for these medium feature maps. Keep the vectorized float path.
    if abs(fs - ts) < 1e-9: return x
    return np.round(x.astype(np.float32) * (fs / ts)).clip(-128, 127).astype(np.int8)

def cat(a, a_sc, b, b_sc, u_sc):
    t0 = time.time()
    r = np.concatenate([rescale_int8(a, a_sc, u_sc), rescale_int8(b, b_sc, u_sc)], axis=0)
    _gp('cat', t0); return r

def resid_cat(cv1, cv1_sc, mc2, mc2_sc, cv2, cv2_sc, u_sc):
    """C3 bottleneck output: concat([cv1 + mc2 (residual add), cv2]) at unified scale.
    The inverted-residual bottleneck adds the shortcut cv1 to the projected mc2
    (verified: L3+L8 @ isc5 == golden L05_in[:32] exactly). All rescaled to u_sc."""
    t0 = time.time()
    bott = np.clip(rescale_int8(cv1, cv1_sc, u_sc).astype(np.int16)
                   + rescale_int8(mc2, mc2_sc, u_sc).astype(np.int16),
                   -128, 127).astype(np.int8)
    r = np.concatenate([bott, rescale_int8(cv2, cv2_sc, u_sc)], axis=0)
    _gp('resid_cat', t0); return r

def upsample2x(x):
    t0 = time.time()
    r = x.repeat(2, axis=1).repeat(2, axis=2)
    _gp('upsample', t0); return r

def maxpool5(x):
    t0 = time.time()
    C, H, W = x.shape
    xp = np.pad(x, ((0,0),(2,2),(2,2)), constant_values=-128)
    out = xp[:, 0:H, 0:W].copy()
    for dh in range(5):
        for dw in range(5):
            out = np.maximum(out, xp[:, dh:dh+H, dw:dw+W])
    _gp('maxpool', t0); return out

def dequant(x, sc):
    return x.astype(np.float32) * sc


def make_runner(meta, req, weights, bias, verify=False):
    osc = lambda lid: meta[f"layer_{lid:02d}"]["output_scale"]
    isc = lambda lid: meta[f"layer_{lid:02d}"]["input_scale"]

    def conv(lid, x, relu=True):
        k = f"layer_{lid:02d}"; m = meta[k]
        if k not in req:   # FP32 no-requant: 42/49/56 (leaky-0.1), 62 (detect out, no relu)
            return float_conv(x, weights[k], bias[k],
                              m["input_scale"], m["weight_scale_per_channel"],
                              stride=m["stride"][0], pad=m["padding"][0],
                              is_depthwise=bool(m["is_depthwise"]),
                              apply_relu=(lid != 62))
        r = relu and (lid not in FP32_NO_LEAKY)
        out = quant_conv(x, weights[k], bias[k],
                         np.array(req[k]["M0"], np.uint16), np.array(req[k]["shift"], np.uint8),
                         stride=m["stride"][0], pad=m["padding"][0],
                         is_depthwise=bool(m["is_depthwise"]), apply_relu=r)
        if verify:
            gp = f"{GOLD}/L{lid:02d}_out_CHW.bin"
            if os.path.exists(gp):
                shp = json.load(open(f"{GOLD}/layer_shapes.json")).get(k, {}).get("out")
                if shp and tuple(out.shape) == tuple(shp):
                    g = np.fromfile(gp, np.int8).reshape(shp)
                    d = np.abs(out.astype(np.int16) - g.astype(np.int16))
                    print(f"    L{lid:02d} {str(tuple(out.shape)):16s} within1={100*np.mean(d<=1):.1f}% off>1={100*np.mean(d>1):.2f}%")
        return out
    return conv, osc, isc


def run_graph(img_x, conv, osc, isc):
    C1 = lambda lid, x, relu=True: conv(lid, x, relu)
    C3 = lambda lid, x: conv(lid, x, True)
    DW = lambda lid, x: conv(lid, x, True)

    # BACKBONE
    x0 = C3(0, img_x); x1 = C3(1, x0); x2 = C3(2, x1)
    cv1=C1(3,x2)
    mc1=C1(6,cv1); mdw=DW(7,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(4,x2)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(8,mdw)
    if os.environ.get('DUMP_PREFIX'):
        np.save(f"{os.environ['DUMP_PREFIX']}_mc1_6.npy", np.asarray(mc1))   # DW7 input (clean?)
        np.save(f"{os.environ['DUMP_PREFIX']}_mdw_7.npy", np.asarray(mdw))   # DW7 output (first DW)
    x3 = C1(5, resid_cat(cv1, osc(3), mc2, osc(8), cv2, osc(4), isc(5)))
    x4 = C3(9, x3)
    cv1=C1(10,x4)
    mc1=C1(13,cv1); mdw=DW(14,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(11,x4)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(15,mdw)
    P3 = C1(12, resid_cat(cv1, osc(10), mc2, osc(15), cv2, osc(11), isc(12)))
    x6 = C3(16, P3)
    cv1=C1(17,x6)
    mc1=C1(20,cv1); mdw=DW(21,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(18,x6)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(22,mdw)
    P4 = C1(19, resid_cat(cv1, osc(17), mc2, osc(22), cv2, osc(18), isc(19)))
    x8 = C3(23, P4)
    cv1=C1(24,x8)
    mc1=C1(27,cv1); mdw=DW(28,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(25,x8)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(29,mdw)
    P5bb = C1(26, resid_cat(cv1, osc(24), mc2, osc(29), cv2, osc(25), isc(26)))
    # SPPF
    s1 = C1(30, P5bb)
    m1 = maxpool5(s1); m2 = maxpool5(m1); m3 = maxpool5(m2)
    P5 = C1(31, np.concatenate([s1, m1, m2, m3], axis=0))

    # HEAD top-down
    neck_p5 = C1(32, P5)
    neck4 = cat(upsample2x(neck_p5), osc(32), P4, osc(19), isc(33))
    cv1=C1(33,neck4)
    mc1=C1(36,cv1); mdw=DW(37,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(34,neck4)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(38,mdw)
    x34 = C1(35, cat(mc2, osc(38), cv2, osc(34), isc(35)))
    neck_p4 = C1(39, x34)
    neck3 = cat(upsample2x(neck_p4), osc(39), P3, osc(12), isc(40))
    cv1=C1(40,neck3)
    mc1=C1(43,cv1); mdw=DW(44,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(41,neck3)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(45,mdw)
    if os.environ.get('DUMP_PREFIX'):
        np.save(f"{os.environ['DUMP_PREFIX']}_mc1_43.npy", np.asarray(mc1))
        np.save(f"{os.environ['DUMP_PREFIX']}_mdw_44.npy", np.asarray(mdw))
        np.save(f"{os.environ['DUMP_PREFIX']}_cv1_40.npy", np.asarray(cv1))
        np.save(f"{os.environ['DUMP_PREFIX']}_neck3.npy", np.asarray(neck3))
    _in42 = cat(mc2, osc(45), cv2, osc(41), isc(42))
    if os.environ.get('DUMP_PREFIX'):
        np.save(f"{os.environ['DUMP_PREFIX']}_in42.npy", np.asarray(_in42))
        np.save(f"{os.environ['DUMP_PREFIX']}_mc2_45.npy", np.asarray(mc2))
        np.save(f"{os.environ['DUMP_PREFIX']}_cv2_41.npy", np.asarray(cv2))
    x42 = conv(42, _in42)   # FP32 (float)

    # detect small + C3(46)  — x42 is float → quantize to each consumer's input_scale
    out_small_i = C1(60, quantize_to_int8(x42, isc(60)), relu=False)
    x46 = C3(46, quantize_to_int8(x42, isc(46)))
    neck4b = cat(x46, osc(46), neck_p4, osc(39), isc(47))
    cv1=C1(47,neck4b)
    mc1=C1(50,cv1); mdw=DW(51,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(48,neck4b)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(52,mdw)
    x49 = conv(49, cat(mc2, osc(52), cv2, osc(48), isc(49)))   # FP32 (float)

    # detect medium + C3(53)
    out_medium_i = C1(61, quantize_to_int8(x49, isc(61)), relu=False)
    x53 = C3(53, quantize_to_int8(x49, isc(53)))
    neck5b = cat(x53, osc(53), neck_p5, osc(32), isc(54))
    cv1=C1(54,neck5b)
    mc1=C1(57,cv1); mdw=DW(58,mc1)   # DWASYNC: DW thread bắt đầu ở đây
    cv2=C1(55,neck5b)                      # HW chạy song song với DW-CPU (reorder thuần dataflow)
    mc2=C1(59,mdw)
    x56 = conv(56, cat(mc2, osc(59), cv2, osc(55), isc(56)))   # FP32 (float)

    # detect large — layer 62 is FP32 (float_conv) → already float
    out_large = conv(62, quantize_to_int8(x56, isc(62)))

    _dp = os.environ.get('DUMP_PREFIX')
    if _dp:
        for nm, arr in [('x42', x42), ('x49', x49), ('x56', x56),
                        ('osmall', dequant(out_small_i, osc(60))),
                        ('omed', dequant(out_medium_i, osc(61))), ('olarge', out_large)]:
            np.save(f'{_dp}_{nm}.npy', np.asarray(arr))
        print(f'[DUMP] saved intermediates -> {_dp}_*.npy')
    return (dequant(out_small_i, osc(60)),    # 60,61 int8 → dequant by output_scale
            dequant(out_medium_i, osc(61)),
            out_large)                         # 62 float already


def _nms(boxes, scores, iou):
    x1,y1,x2,y2 = boxes.T; areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        i_ou=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order=order[1:][i_ou<iou]
    return keep

def _nms_classaware(boxes, scores, cls, iou):
    """NMS per class (empty/occupied separately) — keeps an empty slot next to an occupied
    one (different class, never suppressed against each other), only drops same-class dups."""
    keep=[]
    for c in np.unique(cls):
        idx=np.where(cls==c)[0]
        k=_nms(boxes[idx], scores[idx], iou)
        keep.extend(idx[k].tolist())
    return keep

_GRID_CACHE = {}   # (H,W,na) -> grid [na*H*W, 2]; constant across frames
_ANCH_CACHE = {}   # (si,H,W) -> anch [na*H*W, 2]
def _sig(x): return 1.0/(1.0+np.exp(-x))   # float32 sigmoid (2× faster than float64)

def decode(outs, conf=CONF_THRESH, nms=NMS_THRESH):
    na=len(ANCHORS[0]); nop=5+NUM_CLASSES; boxes=[]
    _td=time.time()
    for si,(out,st) in enumerate(zip(outs, STRIDES)):
        C,H,W=out.shape
        if C!=na*nop: print(f"[WARN] scale{si}: {C}ch != {na*nop}"); continue
        p=out.reshape(na,nop,H,W).transpose(0,2,3,1).reshape(-1,nop).astype(np.float32)
        # FILTER-EARLY: sigmoid only obj(col4)+cls(cols5:) → conf, threshold BEFORE decoding xy/wh.
        # (old: full sigmoid + xy/wh decode on ALL ~6300 anchors, then filter. Now decode only survivors.)
        sc_all=_sig(p[:,4:5])*_sig(p[:,5:]); scm=sc_all.max(1); clm=sc_all.argmax(1)
        k=scm>conf
        print(f"  scale{si} ({H}x{W}) max={float(scm.max()):.3f} pass={int(k.sum())}")
        if not k.any(): continue
        # cached grid + anchors (constant per H,W)
        gkey=(H,W,na)
        if gkey not in _GRID_CACHE:
            gy,gx=np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
            _GRID_CACHE[gkey]=np.tile(np.stack([gx,gy],-1).reshape(-1,2),(na,1)).astype(np.float32)
        akey=(si,H,W)
        if akey not in _ANCH_CACHE:
            _ANCH_CACHE[akey]=np.repeat(np.array(ANCHORS[si],np.float32),H*W,axis=0)
        grid=_GRID_CACHE[gkey][k]; anch=_ANCH_CACHE[akey][k]
        ps=p[k]                                      # survivors only → decode xy/wh on these
        xy=(_sig(ps[:,:2])*2-0.5+grid)*st
        wh=(_sig(ps[:,2:4])*2)**2*anch
        sc=scm[k]; cl=clm[k]
        x1=xy[:,0]-wh[:,0]/2; y1=xy[:,1]-wh[:,1]/2; x2=xy[:,0]+wh[:,0]/2; y2=xy[:,1]+wh[:,1]/2
        w_,h_=wh[:,0],wh[:,1]; asp=np.where(h_>0,w_/h_,0)
        ok=(w_>=MIN_BOX_PX)&(h_>=MIN_BOX_PX)&(w_<=MAX_BOX_PX)&(h_<=MAX_BOX_PX)&(asp>=MIN_ASPECT)&(asp<=MAX_ASPECT)
        if ok.any():   # VECTORIZED: stack [x1,y1,x2,y2,sc,cl] array per scale (no per-box Python append)
            boxes.append(np.stack([x1[ok],y1[ok],x2[ok],y2[ok],sc[ok],cl[ok].astype(np.float32)],axis=1))
    _gp('decode_pre', _td)
    if not boxes: return []
    allb=np.concatenate(boxes,axis=0)
    _tn=time.time()
    keep=_nms_classaware(allb[:,:4],allb[:,4],allb[:,5].astype(np.int32),nms)[:MAX_DETS]
    if AGN_IOU > 0 and keep:                 # lượt agnostic: xoá duplicate chéo lớp (cùng ô đỗ)
        kb=allb[keep]; ka=_nms(kb[:,:4], kb[:,4], AGN_IOU)
        keep=[keep[i] for i in ka]
    _gp('nms', _tn)
    return [allb[i].tolist() for i in keep]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="run from golden input, check each layer")
    ap.add_argument("--image", default=None)
    args=ap.parse_args()
    print("[*] loading model..."); meta,req,weights,bias=load_model()
    if args.image:
        from PIL import Image
        im=Image.open(args.image).convert("RGB").resize((IMG_SIZE,IMG_SIZE))
        arr=np.asarray(im,np.float32).transpose(2,0,1)/255.0
        img=np.round(arr/INPUT_SCALE).clip(-128,127).astype(np.int8)
    else:
        shp=json.load(open(f"{GOLD}/layer_shapes.json"))["layer_00"]["in"]
        img=np.fromfile(f"{GOLD}/L00_in_CHW.bin",np.int8).reshape(shp)
    print(f"[*] input {img.shape}  (verify={args.verify})")
    conv,osc,isc=make_runner(meta,req,weights,bias,verify=args.verify)
    t0=time.time(); outs=run_graph(img,conv,osc,isc)
    print(f"[*] forward done in {time.time()-t0:.1f}s  scales={[o.shape for o in outs]}")
    dets=decode(outs)
    print(f"[*] {len(dets)} detections:")
    for b in dets[:20]:
        print(f"    cls={CLASS_NAMES[int(b[5])]:14s} conf={b[4]:.3f} box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

if __name__=="__main__":
    main()
