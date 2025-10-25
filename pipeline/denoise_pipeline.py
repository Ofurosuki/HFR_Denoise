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
    def __init__(self, ckpt_path: str, device: str = None, strict: bool = True, mask_expansion: int = 0):
        self.device = torch.device(device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
        self.mask_expansion = mask_expansion

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

    @staticmethod
    def expand_mask(mask: torch.Tensor, expansion_size: int) -> torch.Tensor:
        if expansion_size <= 0:
            return mask

        original_shape = mask.shape
        sequence_length = original_shape[-1]

        reshaped_mask = mask.view(-1, 1, sequence_length).float()

        kernel_size = 2 * expansion_size + 1
        padding = expansion_size

        dilated_reshaped_mask = torch.nn.functional.max_pool1d(
            reshaped_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        dilated_mask = dilated_reshaped_mask.view(original_shape).bool()
        return dilated_mask

    @torch.no_grad()
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    @torch.no_grad()
    def predict_attack_mask(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        pred_cls = logits.argmax(dim=1)
        mask = (pred_cls == self.attack_cls)
        if self.mask_expansion > 0:
            mask = self.expand_mask(mask, self.mask_expansion)
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

    for i in tqdm(range(num_frames), desc="Denoising signals"):
        #print(f"Denoising signal {i + 1}/{num_frames}...")
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
    import glob
    import json
    import blosc2
    import shutil

    parser = argparse.ArgumentParser(description="Denoise LiDAR signals from an .npz file or blosc2 directory.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input .npz file or directory containing blosc2 frames.")
    parser.add_argument('--output-path', type=str, default="denoised_lidar_signals", help="Path to save the output .npz file or blosc2 directory.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to process from the input.")
    parser.add_argument("--mask-expansion", type=int, default=0, help="Number of samples to expand the mask by on each side.")

    args = parser.parse_args()

    defender = DenoisePipeline(ckpt_path=args.ckpt_path, mask_expansion=args.mask_expansion)

    print("Loading Data...")
    signals = []
    offsets = []
    frame_tokens = []
    input_is_dir = False

    if os.path.isfile(args.input_path) and args.input_path.endswith('.npz'):
        data = np.load(args.input_path)
        signals = data['signals']
        offsets = data.get('initial_azimuth_offsets')
    elif os.path.isdir(args.input_path):
        input_is_dir = True
        frame_dirs = sorted(glob.glob(os.path.join(args.input_path, '*')))
        frame_dirs = [d for d in frame_dirs if os.path.isdir(d)]
        
        for frame_dir in tqdm(frame_dirs, desc="Loading frames"):
            signal_path = os.path.join(frame_dir, 'signal.bl2')
            config_path = os.path.join(frame_dir, 'config.json')
            if os.path.exists(signal_path) and os.path.exists(config_path):
                with open(signal_path, 'rb') as f:
                    packed_signal = f.read()
                signal_data = blosc2.unpack_array(packed_signal)
                signals.append(signal_data)
                
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                offsets.append(config_data.get('initial_azimuth_offset'))
                frame_tokens.append(os.path.basename(frame_dir))
        
        signals = np.array(signals)
        if any(o is not None for o in offsets):
            offsets = np.array(offsets)
        else:
            offsets = None
    else:
        raise ValueError("Invalid input path. Must be an .npz file or a directory.")

    if args.num_frames:
        signals = signals[:args.num_frames]
        if offsets is not None:
            offsets = offsets[:args.num_frames]
        if frame_tokens:
            frame_tokens = frame_tokens[:args.num_frames]

    print(f"Loaded {len(signals)} signals.")

    denoised_signals = run_denoising(defender, signals)

    print(f"Saving {len(denoised_signals)} denoised signals to {args.output_path}")

    if args.output_path.endswith('.npz'):
        save_payload = {'signals': denoised_signals}
        if offsets is not None:
            save_payload['initial_azimuth_offsets'] = offsets
        np.savez(args.output_path, **save_payload)
    else: # Save as blosc2 directory
        os.makedirs(args.output_path, exist_ok=True)
        if not frame_tokens:
            frame_tokens = [f"frame_{i:06d}" for i in range(len(denoised_signals))]

        for i, denoised_signal in enumerate(tqdm(denoised_signals, desc="Saving frames")):
            token = frame_tokens[i]
            frame_output_dir = os.path.join(args.output_path, token)
            os.makedirs(frame_output_dir, exist_ok=True)

            # Save denoised signal
            with open(os.path.join(frame_output_dir, 'signal.bl2'), 'wb') as f:
                f.write(blosc2.pack_array(denoised_signal))

            # Copy other files from original directory if input was a directory
            if input_is_dir:
                input_frame_dir = os.path.join(args.input_path, token)
                for filename in ['config.json', 'labels.bl2', 'answer_matrix.bl2']:
                    src_file = os.path.join(input_frame_dir, filename)
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, os.path.join(frame_output_dir, filename))
            else: # Create a minimal config if input was .npz
                config_data = {}
                if offsets is not None and i < len(offsets):
                    config_data['initial_azimuth_offset'] = float(offsets[i])
                with open(os.path.join(frame_output_dir, 'config.json'), 'w') as f:
                    json.dump(config_data, f, indent=2)

    print("Done.")
