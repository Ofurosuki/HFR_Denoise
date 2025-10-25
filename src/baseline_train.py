import sys
sys.path.append("/home/zhang/Project/HFR_Denoise")
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import HistMatrixDataset
from utils.utils import make_collate_fn, validate
from model.baseline_model import DAxisConvBaseline
from tqdm import tqdm


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


# ====== Data Wrapper for Eval ======
class EvalConcatWrapper(nn.Module):
    """
    Slice data in multiple forward pass
    """
    def __init__(self, base_model: nn.Module, chunk_size: int = 8192):
        super().__init__()
        self.base = base_model
        self.chunk_size = int(max(1, chunk_size))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, D) -> (B, C, H, W, D)
        """
        B, H, W, D = x.shape
        N = B * H * W
        x_flat = x.reshape(N, D)

        out_chunks = []
        cs = self.chunk_size
        for s in range(0, N, cs):
            e = min(s + cs, N)
            xs = x_flat[s:e].unsqueeze(1).unsqueeze(1)   # (n, 1, 1, D)
            ys = self.base(xs)                           # (n, C, 1, 1, D)
            ys = ys.squeeze(2).squeeze(2)                # (n, C, D)
            out_chunks.append(ys)

        logits_flat = torch.cat(out_chunks, dim=0)       # (N, C, D)
        C = logits_flat.shape[1]
        logits = logits_flat.view(B, H, W, C, D).permute(0, 3, 1, 2, 4).contiguous()
        return logits


# ====== Chunk Data for Training ======
def train_one_epoch_spatial_microbatch(
    base_model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    chunk_size: int = 4096,
) -> float:

    base_model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(enumerate(data_loader), total=num_batches, desc="Train", ncols=100)
    for batch_idx, batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise RuntimeError("Unexpected batch format from collate_fn.")

        x = x.to(device, non_blocking=True)  # (B, H, W, D)
        y = y.to(device, non_blocking=True)  # (B, H, W, D)
        B, H, W, D = x.shape
        N = B * H * W

        optimizer.zero_grad(set_to_none=True)

        x_flat = x.reshape(N, D)
        y_flat = y.reshape(N, D)
        cs = int(max(1, chunk_size))
        loss_sum = 0.0

        for s in range(0, N, cs):
            e = min(s + cs, N)
            xs = x_flat[s:e].unsqueeze(1).unsqueeze(1)  # (n, 1, 1, D)
            ys = y_flat[s:e]                            # (n, D)

            with torch.amp.autocast('cuda', enabled=True):
                logits_5d = base_model(xs)              # (n, C, 1, 1, D)
                logits_1d = logits_5d.squeeze(2).squeeze(2)  # (n, C, D)

                # CrossEntropy: (n, C, D) vs (n, D)
                loss_ce = ce_loss(logits_1d, ys)
                # Dice: (n, C, 1, 1, D) vs (n, 1, 1, D)
                loss_dc = dice_loss(logits_5d, ys.unsqueeze(1).unsqueeze(1))

                loss_chunk = loss_ce + loss_dc

            scaler.scale(loss_chunk).backward()
            loss_sum += loss_chunk.detach().item() * (e - s)

        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss_sum / float(N)
        total_loss += batch_loss

        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        del x, y, x_flat, y_flat
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, num_batches)
    pbar.close()
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/dataset/HFR_Denoise", help="HFR dataset root.")
    parser.add_argument("--normalize_max", type=float, default=9.0, help="Per-sample max for normalization; set <=0 to disable.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epoches.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Training learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay norm.")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension of baseline model.")
    parser.add_argument("--use_d_attn", action="store_true", help="Use self-attention at D-axis bottleneck.")
    parser.add_argument("--workers", type=int, default=2, help="Training workers.")
    parser.add_argument("--out-dir", type=str, default="./run_baseline", help="Save path of model weights.")
    parser.add_argument("--chunk_size", type=int, default=16384, help="Micro-batch size over spatial points (H*W) in training.")
    parser.add_argument("--val_chunk_size", type=int, default=16384, help="Micro-batch size over spatial points (H*W) in validation.")
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
        sync_angle=1,
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

    # ======= Baseline Model =======
    base_model = DAxisConvBaseline(
        in_channels=1,
        num_classes=3,
        hidden_dim=args.hidden_dim,
        use_d_attn=args.use_d_attn,
    ).to(device)

    # Wrapper for Eval
    eval_model = EvalConcatWrapper(base_model, chunk_size=args.val_chunk_size).to(device)

    # ======= Loss =======
    class_weights = torch.tensor([1.0, 3.0, 6.0], device=device, dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(classes=(1, 2))

    # ======= Optim/Sch/Scaler =======
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_score = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    # ======= Training Loop =======
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch_spatial_microbatch(
            base_model, train_loader, optimizer, scaler, device, ce_loss, dice_loss, chunk_size=args.chunk_size
        )

        val_loss, metrics = validate(eval_model, val_loader, device, ce_loss, dice_loss)
        scheduler.step()

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
                "model": base_model.state_dict(),
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
