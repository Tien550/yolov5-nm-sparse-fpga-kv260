"""
eval_qat_test.py - Sub-step 1.5: Evaluate QAT model on TEST split
==================================================================
Khac eval_ptq.py:
- Load QAT best.pt (da co activation scales hoc tu QAT)
- Khong can calibration
- Chay tren test split (chua thay khi train)
"""
import torch, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from convert_to_quant import convert_model
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr

FUSED       = "runs/export/pruned_fused.pt"
QAT_CKPT    = "runs/train/qat_3fixes13/weights/best.pt"
DATA        = "data/pklot.yaml"
IMGSZ       = 416
BATCH       = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("  SUB-STEP 1.5: QAT FINAL EVAL ON TEST SPLIT")
    print("=" * 60)
    device = torch.device(DEVICE)

    print(f"\n[1] Load FP32 base: {FUSED}")
    base = torch.load(FUSED, map_location="cpu", weights_only=False)
    model = base["model"].float()

    print(f"[2] Rebuild quant structure")
    convert_model(model)

    print(f"[3] Load QAT weights: {QAT_CKPT}")
    ckpt = torch.load(QAT_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(miss)}  unexpected={len(unexp)}")

    model = model.to(device)
    model.eval()
    if not hasattr(model, "names"):
        data_dict_tmp = check_dataset(DATA)
        model.names = data_dict_tmp["names"]

    print(f"\n[4] Build TEST dataloader")
    data_dict = check_dataset(DATA)
    if "test" not in data_dict or data_dict["test"] is None:
        print("  WARN: test split khong co trong yaml, fallback sang val")
        test_path = data_dict["val"]
    else:
        test_path = data_dict["test"]

    gs = max(int(model.stride.max()) if hasattr(model, "stride") else 32, 32)
    test_loader, _ = create_dataloader(
        test_path, imgsz=IMGSZ, batch_size=BATCH, stride=gs,
        hyp=None, augment=False, cache=False, pad=0.5, rect=True,
        workers=0, prefix=colorstr("test: "), shuffle=False,
    )

    save_dir = Path("runs/val/qat_final_test")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5] Run validation on test split")
    results, _, _ = yolo_val.run(
        data=data_dict,
        model=model,
        dataloader=test_loader,
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
    mp, mr, map50, map5095 = results[0], results[1], results[2], results[3]
    print("\n" + "=" * 60)
    print("  QAT FINAL RESULTS (TEST SPLIT)")
    print("=" * 60)
    print(f"  Precision:     {mp:.4f}")
    print(f"  Recall:        {mr:.4f}")
    print(f"  mAP@0.5:       {map50:.4f}")
    print(f"  mAP@0.5:0.95:  {map5095:.4f}")
    print("=" * 60)

    if map50 >= 0.95:
        print("  PASS Sub-step 1.5 - san sang sang Buoc 2 (Weight Extraction)")
    else:
        print(f"  WARN: mAP {map50:.3f} < 0.95")


if __name__ == "__main__":
    main()