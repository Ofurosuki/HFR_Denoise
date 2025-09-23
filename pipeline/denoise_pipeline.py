import sys
sys.path.append("/home/zhang/Project/HFR_Denoise")
import torch
from model.denoise_model import DenoiseModel
from utils.utils import *
from utils.plot_utils import *


class DenoisePipeline:
    """
    HFR Denoising Pipeline:
      1) Load DenoiseModel
      2) Predict HFR Mask
      3) Mask out Input Data
    """
    def __init__(self, ckpt_path: str, device: str = None, strict: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load Checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        ckpt_args = ckpt.get("args", {}) or {}
        hidden_dim = ckpt_args.get("hidden_dim", 32)
        use_axial_attn = ckpt_args.get("use_axial_attn", False)

        # Initialize Model and Load State
        self.model = DenoiseModel(
            in_channels=1,
            num_classes=3,
            hidden_dim=hidden_dim,
            use_axial_attn=use_axial_attn,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        self.model.eval()
        self.attack_cls = 2

        # Normalize Signals
        ckpt_scale = ckpt.get("scale", None)
        if ckpt_scale is not None:
            self.scale = float(ckpt_scale)
        else:
            self.scale = 9.0

    @torch.no_grad()
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    @torch.no_grad()
    def predict_attack_mask(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        pred_cls = logits.argmax(dim=1)
        mask = (pred_cls == self.attack_cls)
        return mask

    @torch.no_grad()
    def denoise(self, x: torch.Tensor, inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.predict_attack_mask(x)
        out = x if inplace else x.clone()

        if out.dim() == 5:
            out[mask.unsqueeze(1).expand_as(out)] = 0
        else:
            out[mask] = 0
        return out, mask


if __name__ == "__main__":
    defender = DenoisePipeline(ckpt_path="./run/0923_dn_dim32_lat_attn.pt")

    # Prepare Data Here
    signals, labels = load_data("/data2/yoshida/hist_matrix_test/lidar_signal.npz")

    x = signals[0]
    x = torch.from_numpy(x).float().unsqueeze(0)
    x = x.to(defender.device, non_blocking=True)

    clean_x, mask = defender.denoise(x)
    clean_x = clean_x.squeeze(0).detach().cpu().numpy()
    hfr_mask = mask.squeeze(0).detach().cpu().numpy().astype(np.bool_)

    # Visualization
    gt_mask = (labels[0] == 2)
    save_path = "./vis/Denoise_Vis.png"
    orig_x_np = x.squeeze(0).detach().cpu().numpy()

    save_3d_panels(
        orig_data=orig_x_np,
        denoised_data=clean_x,
        gt_mask=gt_mask.astype(bool),
        pred_mask=hfr_mask.astype(bool),
        save_path=save_path,
        scale=(1 / 10, 1 / 5, 3.0),
        quantile=0.999,
        max_points=None,
        s=0.2
    )