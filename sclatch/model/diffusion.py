"""Diffusion schedule for DDPM training.

Linear beta schedule from Ho et al. (2020), scaled by 1000/T
as in RegDiffusion (Zhu & Slonim, 2024).
"""

import torch


class DiffusionSchedule:
    """Linear beta schedule for DDPM.

    Precomputes sqrt(alpha_bar) and sqrt(1 - alpha_bar) for efficient
    forward diffusion q(x_t | x_0).

    Parameters
    ----------
    T : int
        Number of diffusion timesteps (default 5000).
    beta_start : float
        Starting beta value (default 0.0001).
    beta_end : float
        Ending beta value (default 0.02).
    device : str
        Torch device.
    """

    def __init__(self, T=5000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.T = T
        scale = 1000 / T
        betas = torch.linspace(scale * beta_start, scale * beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(alpha_bars).to(device)
        self.sqrt_one_minus = torch.sqrt(1.0 - alpha_bars).to(device)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = (self.sqrt_alpha_bar[t].unsqueeze(-1) * x_0 +
               self.sqrt_one_minus[t].unsqueeze(-1) * noise)
        return x_t, noise

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        return torch.randint(0, self.T, (batch_size,), device=device)

    def to(self, device):
        """Move schedule tensors to device."""
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        return self
