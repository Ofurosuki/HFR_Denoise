import sys
sys.path.append("/home/zhang/Project/HFR_Denoise")
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.denoise_model import DenoiseModel
from dataset.dataset import HistMatrixDataset
from utils.utils import make_collate_fn, validate


# ====== Dice loss ======
class DiceLoss(nn.Module):
    def __init__(self, classes=(1, 2), eps=1e-6):
        super().__init__()
        self.classes = classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W, D)
        target: (B, H, W, D)  int64
        Calculate Dice to (Object & HFR)
        """
        probs = logits.softmax(dim=1)
        total = 0.0
        for cls in self.classes:
            p = probs[:, cls, ...]
            t = (target == cls).float()
            inter = (p * t).sum()
            denom = (p + t).sum()
            dice = (2 * inter + self.eps) / (denom + self.eps)
            total += (1.0 - dice)
        return total / len(self.classes)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DenoiseModel on val split.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--data_root", type=str, default="/home/dataset/HFR_Denoise", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension of denoising model.")
    parser.add_argument("--use_axial_attn", action="store_true", help="Activate self-attention at latent space.")
    parser.add_argument("--normalize_max", type=float, default=9.0, help="Per-sample max for normalization; set <=0 to disable.")
    parser.add_argument("--override", action="store_true", help="Ignore args in ckpt while use given args to override model.")
    args = parser.parse_args()

    # ====== Device ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ====== Load checkpoint ======
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=device)

    # Read hyper-parameters during training (if exists w/o override)
    if (not args.override) and isinstance(ckpt, dict):
        if "args" in ckpt and isinstance(ckpt["args"], dict):
            saved = ckpt["args"]
            args.hidden_dim = saved.get("hidden_dim", args.hidden_dim)
            args.use_axial_attn = saved.get("use_axial_attn", args.use_axial_attn)
        if "normalize_max" in ckpt:
            args.normalize_max = ckpt["normalize_max"]

    # ====== Dataset & DataLoader======
    val_set = HistMatrixDataset(
        root_path=args.data_root,
        split="val",
        dataset_name="nuscenes",
        scan_type="horizontal",
        sync_angle=1,
        transform=None,
    )
    collate = make_collate_fn(args.normalize_max)
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
        collate_fn=collate,
    )
    print(f"Val size: {len(val_set)}")

    # ====== Model ======
    model = DenoiseModel(
        in_channels=1,
        num_classes=3,
        hidden_dim=args.hidden_dim,
        use_axial_attn=args.use_axial_attn,
    ).to(device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

    model.eval()

    # ====== Loss ======
    class_weights = torch.tensor([1.0, 3.0, 6.0], device=device, dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(classes=(1, 2))

    # ====== Evaluate ======
    with torch.no_grad():
        val_loss, metrics = validate(model, val_loader, device, ce_loss, dice_loss)

    mean_iou = 0.5 * (metrics["obj_IoU"] + metrics["atk_IoU"])
    mean_ap = metrics["mAP"]

    print("\n=========== Evaluation on Val ===========")
    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"Checkpoint epoch: {ckpt['epoch']}")
        if "best_score" in ckpt:
            print(f"Checkpoint best mean IoU: {ckpt['best_score']:.4f}")
    print(f"Val Loss     : {val_loss:.4f}")
    print(f"Object IoU   : {metrics['obj_IoU']:.4f}")
    print(f"Attack IoU   : {metrics['atk_IoU']:.4f}")
    print(f"Object mAP   : {metrics['obj_mAP']:.4f}")
    print(f"Attack mAP   : {metrics['atk_MAP'] if 'atk_MAP' in metrics else metrics['atk_mAP']:.4f}")
    print(f"mAP (overall): {mean_ap:.4f}")
    print(f"mean IoU (O/A): {mean_iou:.4f}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
