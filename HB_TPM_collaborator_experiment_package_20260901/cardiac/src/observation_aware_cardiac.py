#!/usr/bin/env python3
"""PyTorch implementation of observation-aware cardiac HB-TPM.

The estimator-facing API deliberately accepts only observed frames. Full-cycle
images are never an argument to ``forward``; full-cycle times are safe design
variables and full-cycle masks are supplied to the training loss outside this
module for source patients only.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CoefficientMode = Literal["hierarchical", "weak", "source_mean"]


def periodic_fourier_basis(phases: Tensor, harmonics: int) -> Tensor:
    """Return ``[1, cos, sin, ...]`` at phases in cycles."""

    columns = [torch.ones_like(phases)]
    for order in range(1, int(harmonics) + 1):
        angle = 2.0 * math.pi * order * phases
        columns.extend([torch.cos(angle), torch.sin(angle)])
    return torch.stack(columns, dim=-1)


class FrameEncoder(nn.Module):
    """Shared 2-D encoder with heteroskedastic latent observations."""

    def __init__(self, latent_dim: int, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(max(1, c // 8), c),
            nn.SiLU(),
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(max(1, c // 4), 2 * c),
            nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(max(1, c // 2), 4 * c),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mean_head = nn.Linear(4 * c, int(latent_dim))
        self.log_variance_head = nn.Linear(4 * c, int(latent_dim))

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        if frames.ndim != 4 or frames.shape[1] != 1:
            raise ValueError(
                "FrameEncoder expects (N,1,H,W); " f"received {tuple(frames.shape)}"
            )
        features = self.backbone(frames).flatten(1)
        mean = self.mean_head(features)
        log_variance = self.log_variance_head(features).clamp(-7.0, 3.0)
        return mean, log_variance


class MaskDecoder(nn.Module):
    """Decode one latent state per phase to an LV-mask logit image."""

    def __init__(self, latent_dim: int, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.project = nn.Linear(int(latent_dim), 4 * c * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(max(1, c // 4), 2 * c),
            nn.SiLU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),
            nn.GroupNorm(max(1, c // 8), c),
            nn.SiLU(),
            nn.ConvTranspose2d(c, max(4, c // 2), 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(4, c // 2), 1, kernel_size=3, padding=1),
        )
        self.channels = 4 * c

    def forward(self, states: Tensor, output_size: tuple[int, int]) -> Tensor:
        if states.ndim != 3:
            raise ValueError(
                "MaskDecoder expects (B,T,L); " f"received {tuple(states.shape)}"
            )
        batch, frames, _ = states.shape
        features = self.project(states.reshape(batch * frames, -1))
        features = features.reshape(batch * frames, self.channels, 4, 4)
        logits = self.decoder(features)
        if logits.shape[-2:] != tuple(output_size):
            logits = F.interpolate(
                logits,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits.reshape(batch, frames, 1, *output_size)


class CardiacObservationAwareHBTPM(nn.Module):
    """Sparse-image encoder + hierarchical periodic trajectory posterior.

    The trajectory coefficients have a learned Gaussian source prior. Encoded
    sparse frames are heteroskedastic Gaussian observations of a periodic latent
    trajectory, so the target coefficient posterior is available in closed form.
    Its mean is also the MAP estimator used for deterministic reconstruction.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        harmonics: int = 3,
        base_channels: int = 24,
        phase_alignment: bool = True,
        isotropic_prior: bool = False,
        maximum_phase_shift: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.harmonics = int(harmonics)
        self.basis_dim = 1 + 2 * self.harmonics
        self.phase_alignment = bool(phase_alignment)
        self.isotropic_prior = bool(isotropic_prior)
        self.maximum_phase_shift = float(maximum_phase_shift)

        self.encoder = FrameEncoder(self.latent_dim, base_channels)
        self.decoder = MaskDecoder(self.latent_dim, base_channels)
        self.prior_mean = nn.Parameter(
            torch.zeros(self.basis_dim, self.latent_dim)
        )
        if self.isotropic_prior:
            self.prior_log_variance = nn.Parameter(torch.zeros(()))
            self.register_parameter("prior_cholesky_unconstrained", None)
        else:
            initial_diagonal = math.log(math.expm1(1.0 - 1e-4))
            raw_cholesky = torch.zeros(
                self.latent_dim, self.basis_dim, self.basis_dim
            )
            raw_cholesky.diagonal(dim1=-2, dim2=-1).fill_(initial_diagonal)
            self.prior_cholesky_unconstrained = nn.Parameter(raw_cholesky)
            self.register_parameter("prior_log_variance", None)
        self.phase_head = nn.Sequential(
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
            nn.SiLU(),
            nn.Linear(2 * self.latent_dim, 1),
        )

    def prior_statistics(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return covariance, precision and log determinant for each latent."""

        identity = torch.eye(
            self.basis_dim,
            dtype=self.prior_mean.dtype,
            device=self.prior_mean.device,
        )
        if self.isotropic_prior:
            assert self.prior_log_variance is not None
            log_variance = self.prior_log_variance.clamp(-8.0, 8.0)
            variance = log_variance.exp()
            covariance = variance * identity[None].expand(
                self.latent_dim, -1, -1
            )
            precision = variance.reciprocal() * identity[None].expand(
                self.latent_dim, -1, -1
            )
            log_determinant = self.basis_dim * log_variance
            return (
                covariance,
                precision,
                log_determinant.expand(self.latent_dim),
            )

        assert self.prior_cholesky_unconstrained is not None
        raw = self.prior_cholesky_unconstrained
        diagonal = F.softplus(raw.diagonal(dim1=-2, dim2=-1)) + 1e-4
        cholesky = torch.tril(raw, diagonal=-1) + torch.diag_embed(diagonal)
        covariance = cholesky @ cholesky.transpose(-1, -2)
        precision = torch.cholesky_solve(
            identity.expand(self.latent_dim, -1, -1), cholesky
        )
        log_determinant = 2.0 * torch.log(diagonal).sum(dim=-1)
        return covariance, precision, log_determinant

    def encode_frames(self, observed_frames: Tensor) -> tuple[Tensor, Tensor]:
        """Encode only explicitly observed frames of shape ``(B,K,1,H,W)``."""

        if observed_frames.ndim != 5 or observed_frames.shape[2] != 1:
            raise ValueError(
                "observed_frames must have shape (B,K,1,H,W); "
                f"received {tuple(observed_frames.shape)}"
            )
        batch, shots = observed_frames.shape[:2]
        mean, log_variance = self.encoder(
            observed_frames.reshape(batch * shots, *observed_frames.shape[2:])
        )
        return (
            mean.reshape(batch, shots, self.latent_dim),
            log_variance.reshape(batch, shots, self.latent_dim),
        )

    def infer_phase(self, encoded_mean: Tensor) -> Tensor:
        """Infer a bounded circular phase offset from the observed set only."""

        if not self.phase_alignment:
            return torch.zeros(
                encoded_mean.shape[0],
                dtype=encoded_mean.dtype,
                device=encoded_mean.device,
            )
        summary = torch.cat(
            [
                encoded_mean.mean(dim=1),
                encoded_mean.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.maximum_phase_shift * torch.tanh(
            self.phase_head(summary).squeeze(-1)
        )

    def infer_trajectory_posterior(
        self,
        encoded_mean: Tensor,
        encoded_log_variance: Tensor,
        observed_times: Tensor,
        phase_offset: Tensor,
        *,
        coefficient_mode: CoefficientMode = "hierarchical",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute coefficient posterior mean/covariance and observed basis.

        Returns mean ``(B,P,L)``, covariance ``(B,L,P,P)``, and the observed
        design matrix ``(B,K,P)``.
        """

        shifted = torch.remainder(observed_times + phase_offset[:, None], 1.0)
        design = periodic_fourier_basis(shifted, self.harmonics)
        batch, _, basis_dim = design.shape
        if basis_dim != self.basis_dim:
            raise RuntimeError("Fourier basis dimension mismatch")

        prior_mean = self.prior_mean
        prior_covariance, prior_precision, _ = self.prior_statistics()
        if coefficient_mode == "weak":
            prior_mean = torch.zeros_like(prior_mean)
            identity = torch.eye(
                self.basis_dim,
                dtype=prior_mean.dtype,
                device=prior_mean.device,
            )
            prior_covariance = 1e4 * identity[None].expand(
                self.latent_dim, -1, -1
            )
            prior_precision = 1e-4 * identity[None].expand(
                self.latent_dim, -1, -1
            )

        if coefficient_mode == "source_mean":
            covariance = prior_covariance.unsqueeze(0).expand(
                batch, -1, -1, -1
            )
            mean = prior_mean.unsqueeze(0).expand(batch, -1, -1)
            return mean, covariance, design
        if coefficient_mode not in ("hierarchical", "weak"):
            raise ValueError(f"unknown coefficient_mode={coefficient_mode}")

        observation_precision = encoded_log_variance.neg().exp()
        data_precision = torch.einsum(
            "bkp,bkq,bkl->blpq",
            design,
            design,
            observation_precision,
        )
        precision = data_precision + prior_precision[None]
        prior_rhs = torch.einsum(
            "lpq,lq->lp", prior_precision, prior_mean.transpose(0, 1)
        )[None]
        data_rhs = torch.einsum(
            "bkp,bkl,bkl->blp",
            design,
            observation_precision,
            encoded_mean,
        )
        rhs = prior_rhs + data_rhs
        cholesky = torch.linalg.cholesky(precision)
        identity = torch.eye(
            self.basis_dim,
            dtype=precision.dtype,
            device=precision.device,
        ).expand(batch, self.latent_dim, -1, -1)
        covariance = torch.cholesky_solve(identity, cholesky)
        mean_latent_first = torch.cholesky_solve(
            rhs.unsqueeze(-1), cholesky
        ).squeeze(-1)
        return mean_latent_first.transpose(1, 2), covariance, design

    def latent_trajectory(
        self,
        coefficient_mean: Tensor,
        coefficient_covariance: Tensor,
        full_times: Tensor,
        phase_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        shifted = torch.remainder(full_times + phase_offset[:, None], 1.0)
        design = periodic_fourier_basis(shifted, self.harmonics)
        mean = torch.einsum("btp,bpl->btl", design, coefficient_mean)
        variance = torch.einsum(
            "btp,blpq,btq->btl",
            design,
            coefficient_covariance,
            design,
        ).clamp_min(0.0)
        return mean, variance, design

    def coefficient_kl(
        self,
        coefficient_mean: Tensor,
        coefficient_covariance: Tensor,
    ) -> Tensor:
        """KL(q(W)|p(W)) averaged over batch and latent coordinates."""

        _, prior_precision, logdet_prior = self.prior_statistics()
        difference = coefficient_mean.transpose(1, 2) - self.prior_mean.T[None]
        trace = torch.einsum(
            "lpq,blqp->bl", prior_precision, coefficient_covariance
        )
        quadratic = torch.einsum(
            "blp,lpq,blq->bl", difference, prior_precision, difference
        )
        posterior_cholesky = torch.linalg.cholesky(coefficient_covariance)
        logdet_posterior = 2.0 * torch.log(
            posterior_cholesky.diagonal(dim1=-2, dim2=-1)
        ).sum(dim=-1)
        kl = 0.5 * (
            logdet_prior[None]
            - logdet_posterior
            - self.basis_dim
            + trace
            + quadratic
        )
        return kl.mean()

    def decode_masks(
        self, trajectory_states: Tensor, output_size: tuple[int, int]
    ) -> Tensor:
        return self.decoder(trajectory_states, output_size)

    def forward(
        self,
        observed_frames: Tensor,
        observed_times: Tensor,
        full_times: Tensor,
        *,
        output_size: tuple[int, int],
        coefficient_mode: CoefficientMode = "hierarchical",
    ) -> dict[str, Tensor]:
        encoded_mean, encoded_log_variance = self.encode_frames(observed_frames)
        phase_offset = self.infer_phase(encoded_mean)
        coefficient_mean, coefficient_covariance, observed_design = (
            self.infer_trajectory_posterior(
                encoded_mean,
                encoded_log_variance,
                observed_times,
                phase_offset,
                coefficient_mode=coefficient_mode,
            )
        )
        latent_mean, latent_variance, full_design = self.latent_trajectory(
            coefficient_mean,
            coefficient_covariance,
            full_times,
            phase_offset,
        )
        logits = self.decode_masks(latent_mean, output_size)
        reconstructed_observations = torch.einsum(
            "bkp,bpl->bkl", observed_design, coefficient_mean
        )
        observation_nll = 0.5 * (
            (encoded_mean - reconstructed_observations).square()
            * encoded_log_variance.neg().exp()
            + encoded_log_variance
        ).mean()
        return {
            "mask_logits": logits,
            "encoded_mean": encoded_mean,
            "encoded_log_variance": encoded_log_variance,
            "reconstructed_observations": reconstructed_observations,
            "coefficient_mean": coefficient_mean,
            "coefficient_covariance": coefficient_covariance,
            "latent_mean": latent_mean,
            "latent_variance": latent_variance,
            "phase_offset": phase_offset,
            "full_design": full_design,
            "observation_nll": observation_nll,
        }

    def sample_mask_probabilities(
        self,
        output: dict[str, Tensor],
        *,
        output_size: tuple[int, int],
        n_samples: int,
    ) -> Tensor:
        """Draw coefficient-posterior mask probabilities ``(S,B,T,1,H,W)``."""

        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        mean = output["coefficient_mean"].transpose(1, 2)
        covariance = output["coefficient_covariance"]
        distribution = torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=covariance,
        )
        sampled = distribution.rsample((int(n_samples),))
        sampled = sampled.permute(0, 1, 3, 2)
        latent = torch.einsum(
            "btp,sbpl->sbtl", output["full_design"], sampled
        )
        sample_count, batch, frames, latent_dim = latent.shape
        logits = self.decode_masks(
            latent.reshape(sample_count * batch, frames, latent_dim),
            output_size,
        )
        return torch.sigmoid(
            logits.reshape(sample_count, batch, frames, 1, *output_size)
        )
