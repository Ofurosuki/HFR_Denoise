import sys
sys.path.append("/home/zhang/Project/HFR_Denoise")
import os
import time
import argparse
from model.denoise_model import DenoiseModel
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset.dataset import HistMatrixDataset
from utils.utils import *


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


# ====== Main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/yoshida/dataset_spaal", help="HFR dataset root.")
    parser.add_argument("--normalize_max", type=float, default=9.0, help="Per-sample max for normalization; set <=0 to disable.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epoches.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Training learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay norm.")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension of denoising model.")
    parser.add_argument("--use_axial_attn", action="store_true", help="Activate self-attention at latent space.")
    parser.add_argument("--workers", type=int, default=2, help="Training workers.")
    parser.add_argument("--out-dir", type=str, default="./run", help="Save path of model weights.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ======= Datasets =======
    train_set = HistMatrixDataset(
        root_path=args.data_root,
        split="train",
        dataset_name="nuscenes",
        scan_type="horizontal",
        sync_angle=0,
        transform=None,
    )
    val_set = HistMatrixDataset(
        root_path=args.data_root,
        split="val",
        dataset_name="nuscenes",
        scan_type="horizontal",
        sync_angle=1,
        transform=None,
    )

    # ======= DataLoader =======
    collate = make_collate_fn(args.normalize_max)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
        collate_fn=collate,
    )
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
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    # ======= Model & Optimizer & Loss =======
    model = DenoiseModel(
        in_channels=1,
        num_classes=3,
        hidden_dim=args.hidden_dim,
        use_axial_attn=args.use_axial_attn,
    ).to(device)

    # CLass Weight: Others/Object/Attack
    class_weights = torch.tensor([1.0, 3.0, 6.0], device=device, dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(classes=(1, 2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_score = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    # ======= Training Loop =======
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, ce_loss, dice_loss)
        val_loss, metrics = validate(model, val_loader, device, ce_loss, dice_loss)
        scheduler.step()

        # Evaluation: Object/Attack Avg. IoU and mAP
        mean_iou = 0.5 * (metrics["obj_IoU"] + metrics["atk_IoU"])
        mean_ap = metrics["mAP"]
        t1 = time.time()

        print(f"[Epoch {epoch:03d}] "
              f"Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  "
              f"Object IoU={metrics['obj_IoU']:.4f}  Attack IoU={metrics['atk_IoU']:.4f}  "
              f"Object mAP={metrics['obj_mAP']:.4f}  Attack mAP={metrics['atk_mAP']:.4f}  mAP={mean_ap:.4f}  "
              f"time={t1 - t0:.1f}s")

        # Save the Best
        if mean_iou > best_score:
            best_score = mean_iou
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "args": vars(args),
                "normalize_max": args.normalize_max,
            }, best_path)
            print(f"Saved best to: {best_path} (mean IoU={best_score:.4f})")

    print("Training finished.")
    print(f"Best mean IoU (Object/Attack) = {best_score:.4f}")
    print(f"Best model path: {best_path}")


if __name__ == "__main__":
    main()
