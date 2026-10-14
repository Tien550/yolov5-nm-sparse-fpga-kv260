"""
calibrate.py - Save calibrated quant state_dict (input cho QAT training)
=========================================================================
Chay 1 lan truoc khi sang train.py (QAT).
Output: runs/quant/quant_calibrated.pt
"""
import torch, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from convert_to_quant import convert_model
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr

FUSED         = "runs/export/pruned_fused.pt"
QUANT_IN      = "runs/quant/quant_state_dict.pt"
QUANT_OUT     = "runs/quant/quant_calibrated.pt"
DATA          = "data/pklot.yaml"
IMGSZ         = 416
BATCH         = 8
CALIB_BATCHES = 25
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("  CALIBRATION - PRODUCE quant_calibrated.pt FOR QAT")
    print("=" * 60)
    device = torch.device(DEVICE)

    print(f"\n[1] Load FP32 fused: {FUSED}")
    ckpt = torch.load(FUSED, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    model.eval()

    print(f"[2] Convert to quant structure")
    converted, failed, dw, n_det = convert_model(model)
    print(f"  Converted={converted}  Failed={failed}  DW={dw}  Detect_skip={n_det}")

    print(f"[3] Load quant state_dict: {QUANT_IN}")
    state = torch.load(QUANT_IN, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(miss)} unexpected={len(unexp)}")
    model.to(device)

    print(f"[4] Calibration: {CALIB_BATCHES} batches x {BATCH} = {CALIB_BATCHES*BATCH} images")
    data_dict = check_dataset(DATA)
    gs = max(int(model.stride.max()) if hasattr(model, "stride") else 32, 32)
    loader, _ = create_dataloader(
        data_dict["train"], imgsz=IMGSZ, batch_size=BATCH, stride=gs,
        hyp=None, augment=False, cache=False, pad=0.0, rect=False,
        workers=0, prefix=colorstr("calib: "), shuffle=True,
    )

    model.train()  # Brevitas thu stats trong train mode
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= CALIB_BATCHES:
                break
            imgs = batch[0].to(device, non_blocking=True).float() / 255.0
            _ = model(imgs)
            if (i + 1) % 5 == 0:
                print(f"  calib {i+1}/{CALIB_BATCHES}")

    print(f"\n[5] Save calibrated state_dict -> {QUANT_OUT}")
    out = {
        "state_dict": model.state_dict(),
        "epoch": -1,
        "best_fitness": 0.0,
    }
    Path(QUANT_OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, QUANT_OUT)
    print(f"  OK - {len(out['state_dict'])} keys saved")
    print("\nDONE. Next: patch train.py + run QAT training")


if __name__ == "__main__":
    main()