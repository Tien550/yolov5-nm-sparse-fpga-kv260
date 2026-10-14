"""
inference_sparse_fp32.py -- FP32 inference với sparse model (bypass INT8).
Dùng để verify detection quality của sparse_best.pt mà không cần PTQ/QAT.

Usage:
  python inference_sparse_fp32.py --image ../datasets/pklot/valid/images/xxx.jpg
  python inference_sparse_fp32.py --image ../datasets/pklot/valid/images/xxx.jpg --save out.png
"""
import torch, sys, warnings, argparse, numpy as np, os
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

CONF_THRESH = 0.25
NMS_THRESH  = 0.12
IMG_SIZE    = 416
CLASS_NAMES = ['space-empty', 'space-occupied']
ANCHORS = [[[11,24],[13,30],[15,26]], [[17,34],[22,35],[30,27]], [[27,44],[37,38],[48,63]]]
STRIDES = [8, 16, 32]

def xywh2xyxy(x):
    y = x.clone()
    y[...,0] = x[...,0]-x[...,2]/2; y[...,1] = x[...,1]-x[...,3]/2
    y[...,2] = x[...,0]+x[...,2]/2; y[...,3] = x[...,1]+x[...,3]/2
    return y

def nms(boxes, scores, iou_thresh):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0,xx2-xx1); h=np.maximum(0,yy2-yy1)
        inter=w*h; iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<=iou_thresh]
    return keep

def decode_output(pred, anchors, stride, img_size):
    """Decode YOLO output tensor to boxes."""
    bs,_,gy,gx=pred.shape; na=len(anchors); nc=pred.shape[1]//na-5
    p=pred.view(bs,na,5+nc,gy,gx).permute(0,1,3,4,2).contiguous()
    p=p.sigmoid()
    grid_y,grid_x=torch.meshgrid(torch.arange(gy),torch.arange(gx),indexing='ij')
    grid=torch.stack([grid_x,grid_y],-1).float().view(1,1,gy,gx,2)
    anchor_t=torch.tensor(anchors,dtype=torch.float).view(1,na,1,1,2)
    xy=(p[...,0:2]*2-0.5+grid)*stride
    wh=(p[...,2:4]*2)**2*anchor_t
    conf=p[...,4:5]; cls_p=p[...,5:]
    boxes=torch.cat([xy,wh],dim=-1).view(-1,4)
    scores=(conf*cls_p).view(-1,nc)
    return boxes,scores

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="runs/sparse/sparse_best.pt")
    ap.add_argument("--image",required=True)
    ap.add_argument("--save",default=None)
    ap.add_argument("--conf",type=float,default=CONF_THRESH)
    args=ap.parse_args()

    print(f"[*] loading {args.model}...")
    ckpt=torch.load(args.model,map_location="cpu",weights_only=False)
    model=ckpt["model"].float().eval()
    ep=ckpt.get("epoch","?"); mAP=ckpt.get("mAP50",0)
    print(f"[*] ep={ep} mAP50={mAP:.4f}")

    # Load image
    try: from PIL import Image
    except: import subprocess; subprocess.run([sys.executable,"-m","pip","install","Pillow","-q"])
    from PIL import Image, ImageDraw
    img=Image.open(args.image).convert("RGB").resize((IMG_SIZE,IMG_SIZE))
    arr=np.array(img,dtype=np.float32)/255.0
    x=torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)

    print(f"[*] running forward...")
    with torch.no_grad():
        out=model(x)

    # fused model: out[0] = [1, 10647, 7] (xywh + conf + cls0 + cls1, already sigmoid)
    pred=out[0] if isinstance(out,(list,tuple)) else out  # [1, N, 7]

    from utils.general import non_max_suppression
    results=non_max_suppression(pred, conf_thres=args.conf, iou_thres=NMS_THRESH,
                                max_det=300)

    dets=[]
    for det in results:  # per image
        if det is not None and len(det):
            for *xyxy, conf, cls in det.tolist():
                dets.append([xyxy[0],xyxy[1],xyxy[2],xyxy[3],conf,int(cls)])

    print(f"[*] {len(dets)} detections (conf>{args.conf}):")
    for d in dets[:10]:
        print(f"    {CLASS_NAMES[d[5]]} conf={d[4]:.3f} box=[{d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}]")
    if len(dets)>10: print(f"    ... (+{len(dets)-10} more)")

    if args.save:
        draw=ImageDraw.Draw(img)
        colors=["#00ff00","#ff4444"]
        for d in dets:
            c=colors[d[5]]
            draw.rectangle([d[0],d[1],d[2],d[3]],outline=c,width=2)
        img.save(args.save)
        print(f"[*] saved → {args.save}")

if __name__=="__main__": main()
