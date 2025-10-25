import numpy as np
from typing import Dict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


# ====== Collate FN ======
def make_collate_fn(normalize_max: float):
    def collate_fn(batch):
        signals = []
        labels = []
        for sample in batch:
            sig = sample["signal"].float()
            lab = sample["labels"].long()
            if normalize_max is not None and normalize_max > 0:
                sig = sig / float(normalize_max)
            signals.append(sig)
            labels.append(lab)
        signals = torch.stack(signals, dim=0)
        labels = torch.stack(labels, dim=0)
        return signals, labels
    return collate_fn


# ====== IoU Calculation ======
@torch.no_grad()
def eval_iou_stats_on_batch(pred_logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate tp/fp/fn for IoU (Object = 1, Attack = 2)
    """
    pred = pred_logits.argmax(dim=1)
    pred = pred.cpu()
    target = target.cpu()

    stats = {}
    for cls, name in [(1, "obj"), (2, "atk")]:
        p = (pred == cls)
        t = (target == cls)
        tp = (p & t).sum().item()
        fp = (p & ~t).sum().item()
        fn = (~p & t).sum().item()
        stats[f"{name}_tp"] = tp
        stats[f"{name}_fp"] = fp
        stats[f"{name}_fn"] = fn
    return stats


def aggregate_stats(sum_stats: Dict[str, float], add_stats: Dict[str, float]):
    for k, v in add_stats.items():
        sum_stats[k] = sum_stats.get(k, 0.0) + float(v)


def finalize_iou(sum_stats: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for name in ["obj", "atk"]:
        tp = sum_stats.get(f"{name}_tp", 0.0)
        fp = sum_stats.get(f"{name}_fp", 0.0)
        fn = sum_stats.get(f"{name}_fn", 0.0)
        iou = tp / (tp + fp + fn + 1e-8)
        out[f"{name}_IoU"] = iou
    return out


# ====== mAP utils ======
def init_map_hist(nbins: int = 256):
    pos_hist = {1: np.zeros(nbins, dtype=np.int64), 2: np.zeros(nbins, dtype=np.int64)}
    neg_hist = {1: np.zeros(nbins, dtype=np.int64), 2: np.zeros(nbins, dtype=np.int64)}
    pos_total = {1: 0, 2: 0}
    return pos_hist, neg_hist, pos_total


@torch.no_grad()
def update_map_hist_from_batch(logits: torch.Tensor, target: torch.Tensor, pos_hist, neg_hist, pos_total, nbins: int = 256):
    probs = torch.softmax(logits, dim=1)  # (B, 3, H, W, D)

    for cls in (1, 2):
        p = probs[:, cls, ...]  # (B, H, W, D), on CUDA
        bins = torch.clamp((p * nbins).long(), 0, nbins - 1)  # bin index in [0, nbins-1]
        mpos = (target == cls)  # bool mask

        pos_counts = torch.bincount(bins[mpos].view(-1), minlength=nbins)
        neg_counts = torch.bincount(bins[~mpos].view(-1), minlength=nbins)

        pos_hist[cls] += pos_counts.cpu().numpy()
        neg_hist[cls] += neg_counts.cpu().numpy()
        pos_total[cls] += int(mpos.sum().item())


def ap_from_hist(pos_hist_1d, neg_hist_1d, pos_total: int) -> float:

    if pos_total == 0:
        return 0.0

    tp = np.cumsum(pos_hist_1d[::-1], dtype=np.float64)
    fp = np.cumsum(neg_hist_1d[::-1], dtype=np.float64)

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(pos_total)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return ap


def aggregate_stats(sum_stats: Dict[str, float], add_stats: Dict[str, float]):
    for k, v in add_stats.items():
        sum_stats[k] = sum_stats.get(k, 0.0) + float(v)


def finalize_stats(sum_stats: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for name, cls in [("obj", 1), ("atk", 2)]:
        tp = sum_stats.get(f"{name}_tp", 0.0)
        fp = sum_stats.get(f"{name}_fp", 0.0)
        fn = sum_stats.get(f"{name}_fn", 0.0)
        iou = tp / (tp + fp + fn + 1e-8)
        f1  = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        out[f"{name}_IoU"] = iou
        out[f"{name}_F1"]  = f1
    return out


# ====== Train and Val ======
def train_one_epoch(model, loader, optimizer, scaler, device, ce_loss, dice_loss=None):
    model.train()
    total_loss = 0.0
    count = 0

    pbar = tqdm(loader, desc="Train", total=len(loader), ncols=100)
    for it, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)  # (B, 3, H, W, D)
            loss_ce = ce_loss(logits, y)
            loss = loss_ce
            if dice_loss is not None:
                loss = loss + 0.5 * dice_loss(logits, y)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


@torch.no_grad()
def validate(model, loader, device, ce_loss, dice_loss=None, nbins: int = 256):
    model.eval()
    total_loss = 0.0
    count = 0

    # IoU & mAP Container
    iou_sum_stats = {}
    pos_hist, neg_hist, pos_total = init_map_hist(nbins)

    pbar = tqdm(loader, desc="Valid", total=len(loader), ncols=100)
    for it, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
            loss_ce = ce_loss(logits, y)
            loss = loss_ce
            if dice_loss is not None:
                loss = loss + 0.5 * dice_loss(logits, y)

        total_loss += loss.item()
        count += 1

        # IoU Aggregation
        batch_iou_stats = eval_iou_stats_on_batch(logits, y)
        aggregate_stats(iou_sum_stats, batch_iou_stats)
        # mAP Aggregation
        update_map_hist_from_batch(logits, y, pos_hist, neg_hist, pos_total, nbins=nbins)

        pbar.set_postfix(val_loss=f"{(total_loss / it):.4f}")

    # Finalize IoU
    iou_metrics = finalize_iou(iou_sum_stats)
    # Finalize mAP
    obj_ap = ap_from_hist(pos_hist[1], neg_hist[1], pos_total[1])
    atk_ap = ap_from_hist(pos_hist[2], neg_hist[2], pos_total[2])
    mAP = 0.5 * (obj_ap + atk_ap)

    # Summarize and Return
    avg_loss = total_loss / max(1, count)
    metrics = {
        "obj_IoU": iou_metrics["obj_IoU"],
        "atk_IoU": iou_metrics["atk_IoU"],
        "obj_mAP": obj_ap,
        "atk_mAP": atk_ap,
        "mAP": mAP,
    }
    return avg_loss, metrics
