"""
ptq_sparse.py -- Post-Training Quantization calibration for the SPARSE model.

Why: sparse model (structured N:16) has ~5x smaller PSUMs than dense → reusing dense
requant (M0/shift/output_scale) squashes outputs → 0 detections. This re-calibrates
activation scales from an FP32 forward pass over calibration images and re-derives
per-channel weight_scale, M0/shift, INT32 bias, and INT8 weights for the sparse model.

Output: runs/quant/extracted_sparse/ (same format as extracted/, drop-in for inference_cpu
and fpga_ispec). Validate: SPARSE_EXTRACTED=... python tools/inference_cpu.py <img>.

Usage:
  python ptq_sparse.py --model runs/sparse/sparse_best.pt --ncalib 32
"""
import torch, sys, json, argparse, warnings, shutil
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import torch.nn as nn

IMG=416

def find_best_shift(M_float):
    """M_float -> (M0 uint16, shift uint5) with M0 in [2^15, 2^16)."""
    if M_float <= 0: return 0, 0
    shift=0; val=M_float
    while val < (1<<15) and shift < 31: val*=2; shift+=1
    while val >= (1<<16) and shift > 0: val/=2; shift-=1
    M0=int(round(val))
    if M0 >= (1<<16): M0=(1<<16)-1
    return M0, shift

def load_calib(n):
    from PIL import Image
    d=Path("../datasets/pklot/valid/images")
    files=sorted(d.glob("*.jpg"))[:n]
    xs=[]
    for f in files:
        im=Image.open(f).convert("RGB").resize((IMG,IMG))
        a=np.array(im,dtype=np.float32)/255.0
        xs.append(torch.from_numpy(a.transpose(2,0,1)))
    return torch.stack(xs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="runs/sparse/sparse_best.pt")
    ap.add_argument("--src",default="runs/quant/extracted")
    ap.add_argument("--out",default="runs/quant/extracted_sparse")
    ap.add_argument("--ncalib",type=int,default=32)
    ap.add_argument("--pct",type=float,default=99.99)
    args=ap.parse_args()

    print(f"[1] load sparse model {args.model}")
    ck=torch.load(args.model,map_location="cpu",weights_only=False)
    model=ck["model"].float().eval()
    print(f"    ep={ck.get('epoch','?')} mAP50={ck.get('mAP50',0):.4f}")

    meta=json.load(open(Path(args.src)/"metadata.json"))
    named=dict(model.named_modules())
    sp_mods={n:m for n,m in named.items() if isinstance(m,nn.Conv2d)}

    # output_scale must reflect the POST-ACTIVATION int8 tensor that feeds the next layer.
    # YOLOv5 Conv block = conv2d + (Identity bn) + LeakyReLU. Hook the BLOCK (parent) output
    # for post-leaky range; bare detect convs (no leaky) hook the conv2d directly.
    out_mod={}
    for n in sp_mods:
        parent=named.get(n.rsplit(".",1)[0]) if "." in n else None
        if parent is not None and isinstance(getattr(parent,"act",None), nn.LeakyReLU):
            out_mod[n]=parent
        else:
            out_mod[n]=sp_mods[n]

    # ── calibration: input on conv2d, output on (post-act) block ──
    print(f"[2] calibrate {args.ncalib} images (pct={args.pct}, post-leaky output)")
    in_max={}; out_max={}
    handles=[]
    def mk_in(name):
        def h(mod,inp,out):
            iv=np.percentile(inp[0].detach().abs().flatten().numpy(),args.pct)
            in_max[name]=max(in_max.get(name,0),float(iv))
        return h
    def mk_out(name):
        def h(mod,inp,out):
            ov=np.percentile(out.detach().abs().flatten().numpy(),args.pct)
            out_max[name]=max(out_max.get(name,0),float(ov))
        return h
    for n,m in sp_mods.items():
        handles.append(m.register_forward_hook(mk_in(n)))
        handles.append(out_mod[n].register_forward_hook(mk_out(n)))
    x=load_calib(args.ncalib)
    with torch.no_grad():
        for i in range(x.shape[0]):
            _=model(x[i:i+1])
    for h in handles: h.remove()
    print(f"    captured {len(in_max)} conv layers")

    # ── build extracted_sparse ──
    out=Path(args.out)
    if out.exists(): shutil.rmtree(out)
    (out/"weights_bin").mkdir(parents=True); (out/"bias_bin").mkdir()
    new_meta={}; requant={}
    EPS=1e-12

    for lid in range(63):
        k=f"layer_{lid:02d}"
        if k not in meta: continue
        m=dict(meta[k])  # copy template (name, shape, kernel, is_depthwise, fp32_output, has_bias)
        name=m["name"]
        mod=sp_mods.get(name)
        if mod is None:
            new_meta[k]=m; continue
        w=mod.weight.detach().numpy()                      # [Cout,Cin,kh,kw]
        Cout=w.shape[0]; wf=w.reshape(Cout,-1)
        wsc=np.maximum(np.abs(wf).max(axis=1)/127.0, EPS)  # per-oc weight scale
        isc=max(in_max.get(name,EPS)/127.0, EPS)
        osc=max(out_max.get(name,EPS)/127.0, EPS)

        # INT8 weights
        w_int=np.clip(np.round(wf/wsc[:,None]),-128,127).astype(np.int8).reshape(w.shape)
        (out/"weights_bin"/f"{k}.bin").write_bytes(w_int.tobytes())

        # bias INT32 = round(bias / (isc*wsc))
        if m.get("has_bias",False) and mod.bias is not None:
            b=mod.bias.detach().numpy()
            bias_int=np.round(b/(isc*wsc)).astype(np.int32)
            (out/"bias_bin"/f"{k}.bin").write_bytes(bias_int.tobytes())

        # requant M0/shift (skip fp32-output layers)
        if not m.get("fp32_output",False):
            M0=[]; SH=[]
            for oc in range(Cout):
                M=isc*wsc[oc]/osc
                m0,sh=find_best_shift(M); M0.append(m0); SH.append(sh)
            requant[k]={"M0":M0,"shift":SH}

        m["input_scale"]=isc; m["output_scale"]=osc
        m["weight_scale_per_channel"]=wsc.tolist()
        new_meta[k]=m

    json.dump(new_meta,open(out/"metadata.json","w"))
    json.dump(requant,open(out/"requant_params.json","w"))
    print(f"[3] wrote {out}/  ({len(requant)} requant layers)")
    print(f"    validate: SPARSE_EXTRACTED={out} python ../tools/inference_cpu.py <img>")

if __name__=="__main__": main()
