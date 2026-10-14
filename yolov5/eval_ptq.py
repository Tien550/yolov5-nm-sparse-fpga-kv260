"""
eval_ptq.py - Sub-step 1.3: Calibrate + Evaluate PTQ baseline mAP
==================================================================
Pipeline:
  1. Load FP32 fused model
  2. convert_model() -> rebuild quant structure (in-place)
  3. Load quant_state_dict.pt
  4. CALIBRATION: forward ~200 train images trong train() mode
                  -> Brevitas collect activation statistics
  5. Switch eval() -> activation scales freeze
  6. Run YOLOv5 val.run() de do mAP

Chay:
    cd D:\yolov5-fpga\yolov5
    python eval_ptq.py
"""
import torch
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from convert_to_quant import convert_model
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr

# ========================== CONFIG ==========================
FUSED_WEIGHTS = "runs/export/pruned_fused.pt"
QUANT_STATE   = "runs/quant/quant_state_dict.pt"
DATA_YAML     = "data/pklot.yaml"
IMGSZ         = 416
BATCH         = 8
CALIB_BATCHES = 25      # 25 * 8 = 200 anh calibration
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================


def main():
    print("=" * 60)
    print("  SUB-STEP 1.3: PTQ CALIBRATION + EVAL")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDevice: {device}")

    # ----- 1. Load FP32 fused model -----
    print(f"\n[1] Loading FP32 fused: {FUSED_WEIGHTS}")
    ckpt_fp32 = torch.load(FUSED_WEIGHTS, map_location="cpu", weights_only=False)
    model = ckpt_fp32["model"].float()
    model.eval()
    print(f"  OK")

    # ----- 2. Rebuild quant structure -----
    print(f"\n[2] Rebuild quant structure (convert_model)...")
    converted, failed, dw, n_detect = convert_model(model)
    print(f"  Converted={converted}  Failed={failed}  DW={dw}  Detect_skip={n_detect}")
    if failed > 0:
        print("  FAIL - co layer convert loi"); return

    # ----- 3. Load quant state_dict -----
    print(f"\n[3] Loading quant state_dict: {QUANT_STATE}")
    state = torch.load(QUANT_STATE, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"  (First few missing: {missing[:3]})")

    model.to(device)

    # ----- 4. Calibration -----
    print(f"\n[4] Calibration ({CALIB_BATCHES} batches x {BATCH} = {CALIB_BATCHES*BATCH} images)")
    data_dict = check_dataset(DATA_YAML)
    train_path = data_dict["train"]

    # gs = grid size, default stride 32
    gs = max(int(model.stride.max()) if hasattr(model, "stride") else 32, 32)

    calib_loader, _ = create_dataloader(
        train_path,
        imgsz=IMGSZ,
        batch_size=BATCH,
        stride=gs,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.0,
        rect=False,
        workers=0,           # Windows-safe
        prefix=colorstr("calib: "),
        shuffle=True,
    )

    # QUAN TRONG: train() mode de Brevitas thu stats, nhung KHONG backward
    model.train()
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(calib_loader):
            if i >= CALIB_BATCHES:
                break
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            _ = model(imgs)
            if (i + 1) % 5 == 0:
                print(f"  calib batch {i+1}/{CALIB_BATCHES}")

    print("  Calibration done -> activation scales collected")

    # ----- 5. Eval mode (freeze scales) -----
    model.eval()

    # ----- 6. Run YOLOv5 validation -----
    print(f"\n[5] Running YOLOv5 val.run() on val split...")
    if not hasattr(model, "names"):
        model.names = data_dict["names"]

    # Tao val dataloader thu cong (vi val.run skip tao loader khi co model)
    val_path = data_dict["val"]
    val_loader, _ = create_dataloader(
        val_path,
        imgsz=IMGSZ,
        batch_size=BATCH,
        stride=gs,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.5,
        rect=True,
        workers=0,
        prefix=colorstr("val: "),
        shuffle=False,
    )

    from pathlib import Path as _P
    save_dir = _P("runs/val/ptq_baseline")
    save_dir.mkdir(parents=True, exist_ok=True)

    results, _, _ = yolo_val.run(
        data=data_dict,
        model=model,
        dataloader=val_loader,
        save_dir=save_dir,
        imgsz=IMGSZ,
        batch_size=BATCH,
        device=device,
        workers=0,
        half=False,
        plots=False,
        save_json=False,
        verbose=True,
    )

    # results = (mp, mr, map50, map, *loss)
    mp, mr, map50, map_5095 = results[0], results[1], results[2], results[3]
    print("\n" + "=" * 60)
    print("  PTQ BASELINE RESULTS")
    print("=" * 60)
    print(f"  Precision:    {mp:.4f}")
    print(f"  Recall:       {mr:.4f}")
    print(f"  mAP@0.5:      {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map_5095:.4f}")
    print("=" * 60)

    if   map50 > 0.85: verdict = "EXCELLENT - QAT chi can fine-tune nhe"
    elif map50 > 0.70: verdict = "OK - QAT se recover duoc"
    else:              verdict = "LOW - kiem tra lai convert/calibration"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()