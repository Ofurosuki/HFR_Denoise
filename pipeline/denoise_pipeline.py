import sys
import time
import os
# Add the parent directory of 'pipeline' (which is HFR_Denoise) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
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


def run_denoising(defender: DenoisePipeline, signals: np.ndarray) -> np.ndarray:
    """
    Denoises a batch of LiDAR signals.
    """
    clean_xs = []
    total_inference_time = 0.0
    num_frames = len(signals)

    for i in range(num_frames):
        print(f"Denoising signal {i + 1}/{num_frames}...")
        x_np = signals[i]
        x = torch.from_numpy(x_np).float().unsqueeze(0)
        x = x.to(defender.device, non_blocking=True)

        inference_start_time = time.time()
        clean_x, _ = defender.denoise(x)
        inference_end_time = time.time()
        total_inference_time += (inference_end_time - inference_start_time)

        clean_x_np = clean_x.squeeze(0).detach().cpu().numpy()
        clean_xs.append(clean_x_np)
    
    if num_frames > 0:
        avg_inference_time = total_inference_time / num_frames
        print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")

    return np.array(clean_xs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Denoise LiDAR signals from an .npz file.")
    parser.add_argument("--input-npz", type=str, required=True, help="Path to the input .npz file containing 'signals'.")
    parser.add_argument("--output-npz", type=str, default="denoised_lidar_signals_batch.npz", help="Path to save the output .npz file.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to process from the input file.")

    args = parser.parse_args()

    defender = DenoisePipeline(ckpt_path=args.ckpt_path)

    print("Loading Data...")
    data = np.load(args.input_npz)
    signals = data['signals']
    offsets = data.get('initial_azimuth_offsets')
    
    if args.num_frames:
        signals = signals[:args.num_frames]
        if offsets is not None:
            offsets = offsets[:args.num_frames]

    print(f"Loaded {len(signals)} signals.")

    denoised_signals = run_denoising(defender, signals)

    print(f"Saving {len(denoised_signals)} denoised signals to {args.output_npz}")
    
    save_payload = {'signals': denoised_signals}
    if offsets is not None:
        save_payload['initial_azimuth_offsets'] = offsets
        
    np.savez(args.output_npz, **save_payload)

    print("Done.")