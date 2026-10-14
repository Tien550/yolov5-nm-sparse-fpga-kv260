"""fp32_detect_count.py -- đếm số detection FP32 (NMS chuẩn) cho 1 model + 1 ảnh.
So FP32 vs INT8 để tách lỗi train (FP32 đã over-detect) hay PTQ (chỉ INT8 hỏng).
Usage: venv python fp32_detect_count.py <model.pt> <image.jpg> [conf] [iou]
"""
import sys, warnings, torch, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
from utils.general import non_max_suppression
from PIL import Image

mp, imgp = sys.argv[1], sys.argv[2]
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
iou  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12

ck = torch.load(mp, map_location="cpu", weights_only=False)
model = ck["model"].float().eval()
print(f"model {mp} mAP50={ck.get('mAP50')}")

im = Image.open(imgp).convert("RGB").resize((416, 416))
x = torch.from_numpy(np.array(im, np.float32).transpose(2, 0, 1)[None] / 255.0)
with torch.no_grad():
    pred = model(x)[0]
# thống kê objectness raw (trước NMS)
raw = model(x)[0] if False else pred
obj = torch.sigmoid(raw[..., 4])
print(f"  objectness raw: mean={obj.mean():.4f} max={obj.max():.4f} >0.25: {(obj>0.25).sum().item()} / {obj.numel()}")
det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, agnostic=False)[0]
print(f"  FP32 detections (conf={conf} iou={iou}): {len(det)}")
