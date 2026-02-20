import sys
import time
import os
import argparse
import glob
import json
import blosc2
import shutil
from tqdm import tqdm

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

    for i in tqdm(range(num_frames), desc="Denoising signals", leave=False):
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
    parser = argparse.ArgumentParser(description="Denoise LiDAR signals from a blosc2 directory.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the directory containing blosc2 frames.")
    parser.add_argument('--output-path', type=str, default="denoised_lidar_signals", help="Path to save the output blosc2 directory.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to process from the input.")
    parser.add_argument("--mask-expansion", type=int, default=0, help="Number of samples to expand the mask by on each side.")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of frames to process in a single batch to control memory usage.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        raise ValueError(f"Input path is not a valid directory: {args.input_path}")

    defender = DenoisePipeline(ckpt_path=args.ckpt_path, mask_expansion=args.mask_expansion)

    print("Gathering frame information...")
    all_frame_info = []  # List of (signal_path, config_path, token, aux_dir_path)
    is_nested_structure = False

    # First, recursively search for directory-based frames (containing 'signal.bl2')
    signal_bl2_paths = sorted(glob.glob(os.path.join(args.input_path, '**', 'signal.bl2'), recursive=True))

    if signal_bl2_paths:
        is_nested_structure = True
        print(f"Found {len(signal_bl2_paths)} 'signal.bl2' files (directory-based structure).")
        for signal_path in signal_bl2_paths:
            frame_dir = os.path.dirname(signal_path)
            config_path = os.path.join(frame_dir, 'config.json')
            token = os.path.basename(frame_dir)
            all_frame_info.append((signal_path, config_path, token, frame_dir))
    else:  # If no 'signal.bl2' found, recursively search for any '.bl2' files
        all_bl2_paths = sorted(glob.glob(os.path.join(args.input_path, '**', '*.bl2'), recursive=True))
        if all_bl2_paths:
            is_nested_structure = False
            print(f"Found {len(all_bl2_paths)} '.bl2' files (file-based structure).")
            for signal_path in all_bl2_paths:
                config_path = os.path.splitext(signal_path)[0] + '.json'
                token = os.path.splitext(os.path.basename(signal_path))[0]
                # In file-based structure, the aux_dir is the file's own directory
                all_frame_info.append((signal_path, config_path, token, os.path.dirname(signal_path)))

    if not all_frame_info:
        print("No signal frames found in the input path. Exiting.")
        exit()

    if args.num_frames:
        all_frame_info = all_frame_info[:args.num_frames]

    batch_size = args.batch_size if args.batch_size > 0 else len(all_frame_info)
    print(f"Found {len(all_frame_info)} total frames. Processing in batches of {batch_size}.")

    os.makedirs(args.output_path, exist_ok=True)

    for i in tqdm(range(0, len(all_frame_info), batch_size), desc="Processing batches"):
        batch_info = all_frame_info[i: i + batch_size]

        batch_signals, batch_offsets, batch_tokens = [], [], []

        for signal_path, config_path, token, _ in batch_info:
            with open(signal_path, 'rb') as f:
                packed_signal = f.read()
            batch_signals.append(blosc2.unpack_array(packed_signal))
            batch_tokens.append(token)

            offset = None
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                offset = config_data.get('initial_azimuth_offset')
            batch_offsets.append(offset)

        denoised_signals_batch = run_denoising(defender, np.array(batch_signals))

        for j, denoised_signal in enumerate(denoised_signals_batch):
            token = batch_tokens[j]
            offset = batch_offsets[j]
            _, _, _, original_aux_dir = batch_info[j]

            frame_output_dir = os.path.join(args.output_path, token)
            os.makedirs(frame_output_dir, exist_ok=True)

            with open(os.path.join(frame_output_dir, 'signal.bl2'), 'wb') as f:
                f.write(blosc2.pack_array(denoised_signal))

            config_to_save = {}
            if offset is not None:
                config_to_save['initial_azimuth_offset'] = float(offset)
            with open(os.path.join(frame_output_dir, 'config.json'), 'w') as f:
                json.dump(config_to_save, f, indent=2)

            if is_nested_structure:
                for filename in ['labels.bl2', 'answer_matrix.bl2']:
                    src_file = os.path.join(original_aux_dir, filename)
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, frame_output_dir)

    print("Done.")
