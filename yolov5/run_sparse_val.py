import torch, sys, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import val as yolo_val
from utils.general import check_dataset, colorstr
from utils.dataloaders import create_dataloader

ckpt = torch.load("runs/sparse/sparse_best.pt", map_location="cpu", weights_only=False)
model = ckpt["model"].float().eval()
ep = ckpt.get("epoch", "?"); m = ckpt.get("mAP50", "?")
print(f"Model: ep={ep}, stored_mAP50={m:.4f}")

dd = check_dataset("data/pklot.yaml")
if not hasattr(model, "names"): model.names = dd["names"]
gs = max(int(model.stride.max()), 32)

vl, _ = create_dataloader(dd["val"], imgsz=416, batch_size=16, stride=gs,
    hyp=None, augment=False, cache=False, pad=0.5, rect=True,
    workers=0, prefix=colorstr("val: "), shuffle=False)

r, _, _ = yolo_val.run(data=dd, model=model, dataloader=vl, imgsz=416,
    batch_size=16, device="cpu", workers=0, half=False,
    plots=False, save_json=False, verbose=False)

print(f"RESULT: mAP50={r[2]:.4f} mAP50-95={r[3]:.4f} P={r[0]:.4f} R={r[1]:.4f}")
print(f"Drop vs dense(0.9923): {(0.9923-r[2])*100:.2f}pp")
