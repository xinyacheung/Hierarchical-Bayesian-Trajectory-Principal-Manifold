#!/usr/bin/env python3
"""Verify PyTorch/CUDA and one HB-TPM forward/backward pass without data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Sequence

import torch
import torch.nn.functional as F


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARDIAC_ROOT))

from src.observation_aware_cardiac import CardiacObservationAwareHBTPM  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Permit CPU/MPS for local engineering checks; cluster runs should omit it.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_non_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.allow_non_cuda:
        device = torch.device("cpu")
    else:
        raise SystemExit(
            "CUDA is not available. Check the allocated GPU node, NVIDIA driver, "
            "and CUDA-enabled PyTorch installation."
        )

    torch.manual_seed(20260825)
    model = CardiacObservationAwareHBTPM(
        latent_dim=8,
        harmonics=3,
        base_channels=8,
    ).to(device)
    observed_frames = torch.randn(2, 5, 1, 64, 64, device=device)
    observed_times = torch.tensor(
        [[0.00, 0.20, 0.40, 0.60, 0.80], [0.05, 0.25, 0.45, 0.65, 0.85]],
        device=device,
    )
    full_times = (
        torch.arange(16, dtype=torch.float32, device=device)[None].repeat(2, 1)
        / 16.0
    )
    truth = torch.rand(2, 16, 1, 64, 64, device=device) > 0.7
    output = model(
        observed_frames,
        observed_times,
        full_times,
        output_size=(64, 64),
    )
    loss = F.binary_cross_entropy_with_logits(
        output["mask_logits"], truth.to(torch.float32)
    ) + 1e-3 * model.coefficient_kl(
        output["coefficient_mean"], output["coefficient_covariance"]
    )
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    samples = model.sample_mask_probabilities(
        output,
        output_size=(64, 64),
        n_samples=2,
    )
    if not torch.isfinite(output["mask_logits"]).all() or not finite_gradients:
        raise RuntimeError("non-finite output or gradient in GPU preflight")

    record: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_available": torch.backends.mps.is_available(),
        "forward_shape": list(output["mask_logits"].shape),
        "posterior_sample_shape": list(samples.shape),
        "finite_gradients": bool(finite_gradients),
        "loss": float(loss.detach().cpu()),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "allocated_memory_bytes": torch.cuda.memory_allocated(device),
                "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(record, indent=2, sort_keys=True))
    print("GPU preflight passed.")


if __name__ == "__main__":
    main()
