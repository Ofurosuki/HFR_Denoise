import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def save_3d_panels(orig_data: np.ndarray,
                   denoised_data: np.ndarray,
                   gt_mask: np.ndarray,
                   pred_mask: np.ndarray,
                   save_path: str,
                   scale=(1/10, 1/5, 3.0),
                   quantile: float = 0.999,
                   max_points: int | None = None,
                   s: float = 0.2):

    H, W, D = orig_data.shape
    sW, sD, sH = scale

    def pick_points_from_data(data):
        thr = np.quantile(data, quantile)
        idx = np.argwhere(data > thr)
        if max_points is not None and idx.shape[0] > max_points:
            idx = idx[np.random.choice(idx.shape[0], max_points, replace=False)]
        return idx[:, 0], idx[:, 1], idx[:, 2]

    def pick_points_from_mask(mask):
        idx = np.argwhere(mask)
        if max_points is not None and idx.shape[0] > max_points:
            idx = idx[np.random.choice(idx.shape[0], max_points, replace=False)]
        return idx[:, 0], idx[:, 1], idx[:, 2]

    def setup_axes(ax):
        ax.set_xlim(0, W * sW); ax.set_ylim(0, D * sD); ax.set_zlim(0, H * sH)
        try:
            ax.set_box_aspect((W * sW, D * sD, H * sH))
        except Exception:
            pass
        xticks = np.linspace(0, W, 6)
        yticks = np.linspace(0, D, 6)
        zticks = np.linspace(0, H, 5)
        ax.set_xticks(xticks * sW); ax.set_yticks(yticks * sD); ax.set_zticks(zticks * sH)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v / sW))}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v / sD))}"))
        ax.zaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v / sH))}"))
        ax.set_xlabel("W"); ax.set_ylabel("D"); ax.set_zlabel("H")

    h1, w1, d1 = pick_points_from_data(orig_data)
    h2, w2, d2 = pick_points_from_data(denoised_data)
    h3, w3, d3 = pick_points_from_mask(gt_mask)
    h4, w4, d4 = pick_points_from_mask(pred_mask)

    fig = plt.figure(figsize=(16, 12))
    axes = [
        fig.add_subplot(2, 2, 1, projection='3d'),
        fig.add_subplot(2, 2, 2, projection='3d'),
        fig.add_subplot(2, 2, 3, projection='3d'),
        fig.add_subplot(2, 2, 4, projection='3d'),
    ]
    titles = [
        "Original Data (Top Quantile)",
        "Denoised Data (Top Quantile)",
        "HFR Attack Mask (GT)",
        "Predicted Attack Mask",
    ]
    pts = [(h1, w1, d1), (h2, w2, d2), (h3, w3, d3), (h4, w4, d4)]

    for ax, (hh, ww, dd), title in zip(axes, pts, titles):
        setup_axes(ax)
        if len(hh) == 0:
            ax.text2D(0.15, 0.5, "No points", transform=ax.transAxes)
        else:
            ax.scatter(ww * sW, dd * sD, hh * sH, s=s, alpha=0.85)
        ax.set_title(title)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {save_path}  (scale=W×{sW}, D×{sD}, H×{sH}, max_points={max_points})")
