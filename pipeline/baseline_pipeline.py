import sys
import time
import os
# Add the parent directory of 'pipeline' (which is HFR_Denoise) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from model.baseline_model import DAxisConvBaseline
from utils.utils import *
# from utils.plot_utils import *


class BaselinePipeline:
    """
    HFR Denoising Pipeline using D-Axis Baseline Model:
      1) Load DAxisConvBaseline
      2) Predict HFR Mask
      3) Mask out Input Data
    """
    def __init__(self, ckpt_path: str, device: str = None, strict: bool = True, mask_expansion: int = 0, hidden_dim: int = None, use_d_attn = None, split_h: int = None, use_fp16: bool = False, spatial_chunk_size: int = None):
        self.device = torch.device(device or ("cuda:2" if torch.cuda.is_available() else "cpu"))
        self.mask_expansion = mask_expansion
        self.split_h = split_h  # Number of chunks to split H dimension
        self.use_fp16 = use_fp16  # Use float16 for inference
        self.spatial_chunk_size = spatial_chunk_size  # Chunk size for H*W spatial dimension (e.g., 4096)

        # Load Checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        ckpt_args = ckpt.get("args", {}) or {}
        # Use provided hidden_dim, otherwise try checkpoint, otherwise default to 32
        if hidden_dim is None:
            hidden_dim = ckpt_args.get("hidden_dim", 32)
        # Use provided use_d_attn, otherwise try checkpoint, otherwise default to True
        if use_d_attn is None:
            use_d_attn = ckpt_args.get("use_d_attn", True)

        # Initialize Model and Load State
        self.model = DAxisConvBaseline(
            in_channels=1,
            num_classes=3,
            hidden_dim=hidden_dim,
            use_d_attn=use_d_attn,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        self.model.eval()

        # Convert to float16 if requested
        if self.use_fp16:
            self.model = self.model.half()
            print(f"Model converted to float16 for memory efficiency")

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
        """
        Forward pass with optional spatial chunking to reduce GPU memory usage.
        x: (B, H, W, D)
        return: (B, C, H, W, D)
        """
        # DEBUG: Print input shape
        print(f"[DEBUG] forward_logits input shape: {x.shape}, ndim={x.ndim}")

        # Validate input dimensions (Model expects 4D: B, H, W, D)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, H, W, D), got {x.ndim}D with shape {x.shape}")

        B, H, W, D = x.shape

        # Convert to fp16 if enabled
        if self.use_fp16:
            x = x.half()

        # If spatial_chunk_size is specified, process H*W in chunks (similar to baseline_train.py)
        if self.spatial_chunk_size is not None and self.spatial_chunk_size > 0:
            N = B * H * W
            x_flat = x.reshape(N, D)  # (N, D)

            print(f"[DEBUG] Spatial chunking: N={N} (B={B}, H={H}, W={W}), chunk_size={self.spatial_chunk_size}")

            out_chunks = []
            for s in range(0, N, self.spatial_chunk_size):
                e = min(s + self.spatial_chunk_size, N)
                xs = x_flat[s:e].unsqueeze(1).unsqueeze(1)  # (n, 1, 1, D)

                if s == 0:  # Only print for first chunk
                    print(f"[DEBUG] Processing spatial chunk [0:{e}], shape={xs.shape}")

                logits_chunk = self.model(xs)  # (n, C, 1, 1, D)
                logits_chunk = logits_chunk.squeeze(2).squeeze(2)  # (n, C, D)

                if self.use_fp16:
                    logits_chunk = logits_chunk.float()  # Convert back to fp32

                out_chunks.append(logits_chunk)

                # Free GPU memory
                del xs, logits_chunk
                torch.cuda.empty_cache()

            logits_flat = torch.cat(out_chunks, dim=0)  # (N, C, D)
            C = logits_flat.shape[1]
            logits = logits_flat.view(B, H, W, C, D).permute(0, 3, 1, 2, 4).contiguous()  # (B, C, H, W, D)

            print(f"[DEBUG] Concatenated logits shape: {logits.shape}")
        else:
            # Original full-batch processing
            logits = self.model(x)
            if self.use_fp16:
                logits = logits.float()  # Convert back to fp32

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


def run_denoising(defender: BaselinePipeline, signals: np.ndarray, output_callback=None) -> np.ndarray:
    """
    Denoises a batch of LiDAR signals.

    Args:
        defender: BaselinePipeline instance
        signals: Input signals array
        output_callback: Optional callback(idx, denoised_signal) to process each frame immediately
                        to reduce memory usage

    Returns:
        Array of denoised signals (only if output_callback is None)
    """
    clean_xs = [] if output_callback is None else None
    total_inference_time = 0.0
    num_frames = len(signals)

    for i in tqdm(range(num_frames), desc="Denoising signals"):
        x_np = signals[i]
        if i == 0:  # Only print for first frame
            print(f"[DEBUG] signals[{i}] shape: {x_np.shape}, dtype={x_np.dtype}")

        x = torch.from_numpy(x_np).float().unsqueeze(0)  # (H, W, D) -> (1, H, W, D) - Model expects 4D

        if i == 0:
            print(f"[DEBUG] After unsqueeze: {x.shape} (4D: B, H, W, D)")

        x = x.to(defender.device, non_blocking=True)

        inference_start_time = time.time()
        clean_x, _ = defender.denoise(x)
        inference_end_time = time.time()
        total_inference_time += (inference_end_time - inference_start_time)

        clean_x_np = clean_x.squeeze(0).detach().cpu().numpy()  # (1, H, W, D) -> (H, W, D)

        if output_callback is not None:
            # Immediately process and save, don't accumulate in memory
            output_callback(i, clean_x_np)
        else:
            clean_xs.append(clean_x_np)

    if num_frames > 0:
        avg_inference_time = total_inference_time / num_frames
        print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")

    return np.array(clean_xs) if clean_xs is not None else None


if __name__ == "__main__":
    import argparse
    import glob
    import json
    import blosc2
    import shutil

    parser = argparse.ArgumentParser(description="Denoise LiDAR signals from an .npz file or blosc2 directory using D-Axis Baseline Model.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input .npz file or directory containing blosc2 frames.")
    parser.add_argument('--output-path', type=str, default="denoised_lidar_signals", help="Path to save the output .npz file or blosc2 directory.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to process from the input.")
    parser.add_argument("--mask-expansion", type=int, default=0, help="Number of samples to expand the mask by on each side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension of the model. If not specified, will try to read from checkpoint (default: 32).")
    parser.add_argument("--use-d-attn", action="store_true", help="Use D-axis attention in the model. If not specified, will read from checkpoint (default: True).")
    parser.add_argument("--chunk-size", type=int, default=None, help="Process frames in chunks to reduce memory usage (blosc2 output only). Default: process all at once.")
    parser.add_argument("--spatial-chunk-size", type=int, default=None, help="Spatial chunk size (H*W) for GPU memory efficiency (e.g., 4096). Similar to training chunk_size.")
    parser.add_argument("--split-h", type=int, default=None, help="[Deprecated] Split H dimension into N chunks to reduce GPU memory usage. Use --spatial-chunk-size instead.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision for inference to reduce GPU memory usage.")

    args = parser.parse_args()

    defender = BaselinePipeline(
        ckpt_path=args.ckpt_path,
        mask_expansion=args.mask_expansion,
        hidden_dim=args.hidden_dim,
        use_d_attn=args.use_d_attn,
        split_h=args.split_h,
        use_fp16=args.fp16,
        spatial_chunk_size=args.spatial_chunk_size
    )

    if args.spatial_chunk_size:
        print(f"GPU memory optimization: Spatial chunking enabled with chunk_size={args.spatial_chunk_size}")
    if args.split_h:
        print(f"[Warning] --split-h is deprecated. Consider using --spatial-chunk-size instead.")
        print(f"GPU memory optimization: H dimension will be split into {args.split_h} chunks")
    if args.fp16:
        print(f"GPU memory optimization: Using float16 precision")

    print("Loading Data...")
    signals = []
    offsets = []
    frame_tokens = []
    input_is_dir = False
    frame_dirs_list = []  # For chunk processing

    if os.path.isfile(args.input_path) and args.input_path.endswith('.npz'):
        data = np.load(args.input_path)
        signals = data['signals']
        offsets = data.get('initial_azimuth_offsets')
    elif os.path.isdir(args.input_path):
        input_is_dir = True
        frame_dirs = sorted(glob.glob(os.path.join(args.input_path, '*')))
        frame_dirs = [d for d in frame_dirs if os.path.isdir(d)]

        # Limit number of frames if specified
        if args.num_frames:
            frame_dirs = frame_dirs[:args.num_frames]

        # Check if we should use chunk processing
        use_chunk_processing = (args.chunk_size is not None and
                               not args.output_path.endswith('.npz'))

        if use_chunk_processing:
            # Store frame directories for later chunk processing
            frame_dirs_list = frame_dirs
            print(f"Found {len(frame_dirs_list)} frames (will process in chunks of {args.chunk_size})")
        else:
            # Load all frames into memory (original behavior)
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
            print(f"Loaded {len(signals)} signals.")
    else:
        raise ValueError("Invalid input path. Must be an .npz file or a directory.")

    if not input_is_dir or args.chunk_size is None:
        # Apply num_frames limit if not already applied
        if args.num_frames and len(signals) > 0:
            signals = signals[:args.num_frames]
            if offsets is not None:
                offsets = offsets[:args.num_frames]
            if frame_tokens:
                frame_tokens = frame_tokens[:args.num_frames]

        if len(signals) > 0:
            print(f"Using {len(signals)} signals.")

    # Prepare frame tokens if not already set
    if not frame_tokens and len(signals) > 0:
        frame_tokens = [f"frame_{i:06d}" for i in range(len(signals))]

    # Handle chunk processing for blosc2 directory input
    if frame_dirs_list:
        print(f"Processing {len(frame_dirs_list)} frames in chunks of {args.chunk_size}...")
        os.makedirs(args.output_path, exist_ok=True)

        num_chunks = (len(frame_dirs_list) + args.chunk_size - 1) // args.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * args.chunk_size
            end_idx = min(start_idx + args.chunk_size, len(frame_dirs_list))
            chunk_frame_dirs = frame_dirs_list[start_idx:end_idx]

            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (frames {start_idx}-{end_idx-1})...")

            # Load chunk
            chunk_signals = []
            chunk_offsets = []
            chunk_tokens = []

            for frame_dir in tqdm(chunk_frame_dirs, desc=f"Loading chunk {chunk_idx + 1}"):
                signal_path = os.path.join(frame_dir, 'signal.bl2')
                config_path = os.path.join(frame_dir, 'config.json')
                if os.path.exists(signal_path) and os.path.exists(config_path):
                    with open(signal_path, 'rb') as f:
                        packed_signal = f.read()
                    signal_data = blosc2.unpack_array(packed_signal)
                    chunk_signals.append(signal_data)

                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    chunk_offsets.append(config_data.get('initial_azimuth_offset'))
                    chunk_tokens.append(os.path.basename(frame_dir))

            chunk_signals = np.array(chunk_signals)

            # Callback to save each frame immediately
            def save_chunk_frame_callback(relative_idx, denoised_signal):
                absolute_idx = start_idx + relative_idx
                token = chunk_tokens[relative_idx]
                frame_output_dir = os.path.join(args.output_path, token)
                os.makedirs(frame_output_dir, exist_ok=True)

                # Save denoised signal
                with open(os.path.join(frame_output_dir, 'signal.bl2'), 'wb') as f:
                    f.write(blosc2.pack_array(denoised_signal))

                # Copy other files
                input_frame_dir = chunk_frame_dirs[relative_idx]
                for filename in ['config.json', 'labels.bl2', 'answer_matrix.bl2', 'angles.bl2', 'timestamps.bl2']:
                    src_file = os.path.join(input_frame_dir, filename)
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, os.path.join(frame_output_dir, filename))

            # Process chunk
            run_denoising(defender, chunk_signals, output_callback=save_chunk_frame_callback)

            # Free memory
            del chunk_signals
            import gc
            gc.collect()

        print(f"\nCompleted processing {len(frame_dirs_list)} frames in {num_chunks} chunks.")
        print("Done.")
        sys.exit(0)

    # Memory-efficient processing (non-chunk mode)
    if args.output_path.endswith('.npz'):
        # For npz output, we still need to collect all signals
        print("Processing all frames (npz output requires all signals in memory)...")
        denoised_signals = run_denoising(defender, signals)

        print(f"Saving {len(denoised_signals)} denoised signals to {args.output_path}")
        save_payload = {'signals': denoised_signals}
        if offsets is not None:
            save_payload['initial_azimuth_offsets'] = offsets
        np.savez(args.output_path, **save_payload)

    else:  # Save as blosc2 directory (memory-efficient)
        print("Processing and saving frames one by one (memory-efficient mode)...")
        os.makedirs(args.output_path, exist_ok=True)

        # Callback to save each frame immediately after denoising
        def save_frame_callback(i, denoised_signal):
            token = frame_tokens[i]
            frame_output_dir = os.path.join(args.output_path, token)
            os.makedirs(frame_output_dir, exist_ok=True)

            # Save denoised signal
            with open(os.path.join(frame_output_dir, 'signal.bl2'), 'wb') as f:
                f.write(blosc2.pack_array(denoised_signal))

            # Copy other files from original directory if input was a directory
            if input_is_dir:
                input_frame_dir = os.path.join(args.input_path, token)
                for filename in ['config.json', 'labels.bl2', 'answer_matrix.bl2', 'angles.bl2', 'timestamps.bl2']:
                    src_file = os.path.join(input_frame_dir, filename)
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, os.path.join(frame_output_dir, filename))
            else:  # Create a minimal config if input was .npz
                config_data = {}
                if offsets is not None and i < len(offsets):
                    config_data['initial_azimuth_offset'] = float(offsets[i])
                with open(os.path.join(frame_output_dir, 'config.json'), 'w') as f:
                    json.dump(config_data, f, indent=2)

        # Run denoising with immediate save callback
        run_denoising(defender, signals, output_callback=save_frame_callback)

        print(f"Saved {len(signals)} denoised signals to {args.output_path}")

    print("Done.")
