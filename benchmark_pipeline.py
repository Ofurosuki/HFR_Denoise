#!/usr/bin/env python3
"""
Benchmark script to measure inference time for denoise pipeline
on different LiDAR configurations (VLP-32c and HDL-64E).
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.denoise_pipeline import DenoisePipeline


def create_dummy_data(h, w, d, num_frames=10):
    """
    Create dummy LiDAR histogram data for benchmarking.

    Args:
        h: Height (number of LiDAR channels)
        w: Width (horizontal resolution)
        d: Depth (histogram/time resolution)
        num_frames: Number of frames to generate

    Returns:
        np.ndarray of shape (num_frames, h, w, d)
    """
    print(f"Generating {num_frames} dummy frames with shape ({h}, {w}, {d})...")
    return np.random.randn(num_frames, h, w, d).astype(np.float32)


def benchmark_configuration(
    config_name,
    ckpt_path,
    h,
    w,
    d,
    num_frames,
    device,
    hidden_dim,
    use_axial_attn,
    split_h=None,
    use_fp16=False,
):
    """
    Benchmark a specific LiDAR configuration.

    Args:
        config_name: Name of the configuration (e.g., "VLP-32c")
        ckpt_path: Path to model checkpoint
        h: Height (LiDAR channels)
        w: Width (horizontal resolution)
        d: Depth (time resolution)
        num_frames: Number of frames to process
        device: Device to run on
        hidden_dim: Hidden dimension for model
        use_axial_attn: Axial attention configuration
        split_h: Number of chunks to split H dimension (None = no splitting)
        use_fp16: Use float16 precision
    """
    print("\n" + "=" * 80)
    print(f"Benchmarking: {config_name}")
    print(f"  Input shape: ({h}, {w}, {d})")
    print(f"  Num frames: {num_frames}")
    print(f"  Split H: {split_h}")
    print(f"  FP16: {use_fp16}")
    print("=" * 80)

    # Initialize pipeline
    print("\nInitializing DenoisePipeline...")
    init_start = time.time()
    pipeline = DenoisePipeline(
        ckpt_path=ckpt_path,
        device=device,
        mask_expansion=0,
        hidden_dim=hidden_dim,
        use_axial_attn=use_axial_attn,
        split_h=split_h,
        use_fp16=use_fp16,
    )
    init_time = time.time() - init_start
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Pipeline device: {pipeline.device}")

    # Generate dummy data
    signals = create_dummy_data(h, w, d, num_frames)

    # Warmup run (first inference is often slower)
    print("\nWarmup run...")
    x_warmup = torch.from_numpy(signals[0]).float().unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        _ = pipeline.denoise(x_warmup)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark inference
    print("\nBenchmarking inference...")
    frame_times = []
    total_start = time.time()

    for i in range(num_frames):
        frame_start = time.time()

        # Include data transfer in timing
        x = torch.from_numpy(signals[i]).float().unsqueeze(0).to(pipeline.device)

        # Ensure data transfer is complete before model inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            clean_x, mask = pipeline.denoise(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        frame_time = time.time() - frame_start

        frame_times.append(frame_time)
        print(f"  Frame {i+1}/{num_frames}: {frame_time:.4f} seconds")

    total_time = time.time() - total_start

    # Statistics
    frame_times = np.array(frame_times)
    avg_time = np.mean(frame_times)
    std_time = np.std(frame_times)
    min_time = np.min(frame_times)
    max_time = np.max(frame_times)
    fps = 1.0 / avg_time

    print("\n" + "-" * 80)
    print("Results:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average time per frame: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"  Throughput: {fps:.2f} FPS")
    print("-" * 80)

    return {
        "config_name": config_name,
        "input_shape": (h, w, d),
        "num_frames": num_frames,
        "split_h": split_h,
        "use_fp16": use_fp16,
        "init_time": init_time,
        "total_time": total_time,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark denoise pipeline on different LiDAR configurations"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to benchmark (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden dimension (default: 32)",
    )
    parser.add_argument(
        "--use-axial-attn",
        type=str,
        default="whd",
        help="Axial attention config (default: 'whd')",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        choices=["vlp32", "hdl64", "both"],
        default=["both"],
        help="Which configurations to benchmark (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (optional, .txt or .json)",
    )

    args = parser.parse_args()

    # Determine which configs to run
    if "both" in args.configs:
        configs_to_run = ["vlp32", "hdl64"]
    else:
        configs_to_run = args.configs

    # Configuration parameters
    # Standard sizes from CLAUDE.md and model test bench
    configs = {
        "vlp32": {
            "name": "VLP-32c",
            "h": 32,
            "w": 1800,
            "d": 800,
            "split_h": None,
            "use_fp16": False,
        },
        "vlp32_optimized": {
            "name": "VLP-32c (FP16)",
            "h": 32,
            "w": 1800,
            "d": 800,
            "split_h": None,
            "use_fp16": True,
        },
        "hdl64": {
            "name": "HDL-64E",
            "h": 64,
            "w": 1800,
            "d": 800,
            "split_h": None,
            "use_fp16": False,
        },
        "hdl64_optimized": {
            "name": "HDL-64E (split_h=2, FP16)",
            "h": 64,
            "w": 1800,
            "d": 800,
            "split_h": 2,
            "use_fp16": True,
        },
    }

    results = []

    # Run benchmarks
    for config_key in configs_to_run:
        # Run baseline config
        cfg = configs[config_key]
        result = benchmark_configuration(
            config_name=cfg["name"],
            ckpt_path=args.ckpt_path,
            h=cfg["h"],
            w=cfg["w"],
            d=cfg["d"],
            num_frames=args.num_frames,
            device=args.device,
            hidden_dim=args.hidden_dim,
            use_axial_attn=args.use_axial_attn,
            split_h=cfg["split_h"],
            use_fp16=cfg["use_fp16"],
        )
        results.append(result)

        # Run optimized config for comparison
        if config_key == "vlp32":
            cfg_opt = configs["vlp32_optimized"]
        else:  # hdl64
            cfg_opt = configs["hdl64_optimized"]

        result_opt = benchmark_configuration(
            config_name=cfg_opt["name"],
            ckpt_path=args.ckpt_path,
            h=cfg_opt["h"],
            w=cfg_opt["w"],
            d=cfg_opt["d"],
            num_frames=args.num_frames,
            device=args.device,
            hidden_dim=args.hidden_dim,
            use_axial_attn=args.use_axial_attn,
            split_h=cfg_opt["split_h"],
            use_fp16=cfg_opt["use_fp16"],
        )
        results.append(result_opt)

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Input shape: {result['input_shape']}")
        print(f"  Average time: {result['avg_time']:.4f} ± {result['std_time']:.4f} s")
        print(f"  Throughput: {result['fps']:.2f} FPS")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == ".json":
            import json
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
        else:
            with open(output_path, "w") as f:
                f.write("BENCHMARK RESULTS\n")
                f.write("=" * 80 + "\n\n")
                for result in results:
                    f.write(f"Configuration: {result['config_name']}\n")
                    f.write(f"  Input shape: {result['input_shape']}\n")
                    f.write(f"  Split H: {result['split_h']}\n")
                    f.write(f"  FP16: {result['use_fp16']}\n")
                    f.write(f"  Initialization time: {result['init_time']:.4f} s\n")
                    f.write(f"  Total time: {result['total_time']:.4f} s\n")
                    f.write(f"  Average time per frame: {result['avg_time']:.4f} ± {result['std_time']:.4f} s\n")
                    f.write(f"  Min time: {result['min_time']:.4f} s\n")
                    f.write(f"  Max time: {result['max_time']:.4f} s\n")
                    f.write(f"  Throughput: {result['fps']:.2f} FPS\n")
                    f.write("\n" + "-" * 80 + "\n\n")
            print(f"\nResults saved to {output_path}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
