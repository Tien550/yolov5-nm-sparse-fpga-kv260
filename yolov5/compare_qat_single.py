"""Single-image inference with the DEPLOYED QAT model (= what the FPGA runs), preprocessing MATCHED to
inference_fpga.py (squash-resize 416, /255, conf=0.25, iou=0.20). Prints per-class count + boxes, renders.
Usage: python compare_qat_single.py <image> [out.png]"""
import torch, sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from convert_to_quant import convert_model
from utils.general import non_max_suppression
from PIL import Image, ImageDraw

IMG  = sys.argv[1]
OUT  = sys.argv[2] if len(sys.argv) > 2 else "qat_pc_out.png"
CONF, IOU, SZ = 0.25, 0.20, 416
dev = "cuda" if torch.cuda.is_available() else "cpu"

base = torch.load("runs/export/pruned_fused.pt", map_location="cpu", weights_only=False)
model = base["model"].float(); convert_model(model)
ck = torch.load("runs/train/qat_3fixes13/weights/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck.get("state_dict", ck), strict=False)
model.to(dev).eval()
NAMES = {0: "space-empty", 1: "space-occupied"}

im0 = Image.open(IMG).convert("RGB")
im  = im0.resize((SZ, SZ))                       # SQUASH 416 (matches inference_fpga)
x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(dev)
with torch.no_grad():
    out = model(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
det = non_max_suppression(pred, CONF, IOU, max_det=1000)[0].cpu().numpy()

n_e = int((det[:, 5] == 0).sum()); n_o = int((det[:, 5] == 1).sum())
print(f"[QAT-PC] image={Path(IMG).name}")
print(f"[QAT-PC] total={len(det)}  empty={n_e}  occupied={n_o}  (conf>={CONF}, iou={IOU}, 416-squash)")
for d in det[:8]:
    print(f"    {NAMES[int(d[5])]:14s} conf={d[4]:.3f} box=[{d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}]")

dr = ImageDraw.Draw(im)
for d in det:
    c = (0, 200, 0) if int(d[5]) == 0 else (220, 0, 0)
    dr.rectangle([d[0], d[1], d[2], d[3]], outline=c, width=2)
im.save(OUT)
print(f"[QAT-PC] rendered -> {OUT}")
