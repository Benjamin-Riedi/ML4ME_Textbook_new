"""EngiBench Gen Model Architectures.

This module consolidates all trained generative model architectures used in the EngiBench gen models notebooks
for easy import and reuse in other repositories.

Supported Models:
- CGAN CNN 2D: Conditional GAN with CNN architecture for 2D designs
- LVAE 2D: Least Volume Autoencoder for 2D designs with spectral normalization
- Flow 2D: Normalizing Flow with coupling layers for 2D designs
- Diffusion 2D Conditional: Conditional diffusion model using UNet architecture

Usage:
    from engibench_gen_models import CGANGenerator, CGANDiscriminator, Encoder, SNDecoder, NormalizingFlow, SimpleUNet

Example:
    # Load CGAN generator
    generator = CGANGenerator(latent_dim=32, n_conds=4, design_shape=(50, 100))
    generator.load_state_dict(torch.load('generator.pth'))

    # Load LVAE
    encoder = LVAE_Encoder(latent_dim=10, resize_dimensions=(100, 100))
    decoder = LVAE_SNDecoder(latent_dim=10, design_shape=(50, 100))
    checkpoint = torch.load('lvae.pth')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
"""

from __future__ import annotations

import math
import torch as th
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torchvision import transforms


# ============================================================================
# CGAN CNN 2D Models
# ============================================================================


class CGANGenerator(nn.Module):
    """Simple conditional generator: noise + conditions -> 100x100 design."""

    def __init__(self, latent_dim: int, n_conds: int, design_shape: tuple[int, int]):
        super().__init__()
        self.design_shape = design_shape

        # Separate paths for noise and conditions
        self.z_path = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.c_path = nn.Sequential(
            nn.ConvTranspose2d(
                n_conds, 128, kernel_size=7, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Upsampling blocks: 7x7 -> 13x13 -> 25x25 -> 50x50 -> 100x100
        self.up_blocks = nn.Sequential(
            # 7x7 -> 13x13
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 13x13 -> 25x25
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 25x25 -> 50x50
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 50x50 -> 100x100
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        self.resize = transforms.Resize(design_shape)

    def forward(self, z: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Generate design from noise and conditions.

        Args:
            z: Noise vector (B, latent_dim, 1, 1)
            c: Conditions (B, n_conds, 1, 1)

        Returns:
            Generated design (B, 1, H, W)
        """
        z_feat = self.z_path(z)  # (B, 128, 7, 7)
        c_feat = self.c_path(c)  # (B, 128, 7, 7)
        x = th.cat([z_feat, c_feat], dim=1)  # (B, 256, 7, 7)
        out = self.up_blocks(x)  # (B, 1, 100, 100)
        return self.resize(out)


class CGANDiscriminator(nn.Module):
    """Simple conditional discriminator: design + conditions -> real/fake."""

    def __init__(self, n_conds: int, design_shape: tuple[int, int]):
        super().__init__()
        self.resize = transforms.Resize((100, 100))

        # Image path: 100x100 -> features
        self.img_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 100->50
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 50->25
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Condition path: expand conditions to spatial dimensions
        self.cond_path = nn.Sequential(
            nn.ConvTranspose2d(
                n_conds, 64, kernel_size=25, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Combined path after concatenation
        self.combined = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 25->13
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 13->7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=7, stride=1, padding=0, bias=False),  # 7->1
        )

    def forward(self, img: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Classify design as real or fake given conditions.

        Args:
            img: Design (B, 1, H, W)
            c: Conditions (B, n_conds, 1, 1)

        Returns:
            Validity score (B, 1, 1, 1)
        """
        img = self.resize(img)  # (B, 1, 100, 100)
        img_feat = self.img_path(img)  # (B, 64, 25, 25)
        c_feat = self.cond_path(c)  # (B, 64, 25, 25)
        x = th.cat([img_feat, c_feat], dim=1)  # (B, 128, 25, 25)
        return self.combined(x)  # (B, 1, 1, 1)


# ============================================================================
# LVAE 2D Models
# ============================================================================


class LVAE_Encoder(nn.Module):
    """Simple CNN encoder: 100x100 -> latent vector."""

    def __init__(
        self, latent_dim: int, resize_dimensions: tuple[int, int] = (100, 100)
    ):
        super().__init__()
        self.resize_in = transforms.Resize(resize_dimensions)

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 100→50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 50→25
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 25→13
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 13→7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7x7 conv produces (B, latent_dim, 1, 1) -> flatten to (B, latent_dim)
        self.to_latent = nn.Conv2d(
            512, latent_dim, kernel_size=7, stride=1, padding=0, bias=True
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Encode image to latent vector."""
        x = self.resize_in(x)  # (B,1,100,100)
        h = self.features(x)  # (B,512,7,7)
        return self.to_latent(h).flatten(1)  # (B,latent_dim)


class LVAE_SNDecoder(nn.Module):
    """Spectral normalized CNN decoder: latent vector -> 100x100 image.

    Uses spectral normalization on all linear and convolutional layers
    to enforce 1-Lipschitz bound, which helps stabilize training and
    provides better volume loss behavior.
    """

    def __init__(self, latent_dim: int, design_shape: tuple[int, int]):
        super().__init__()
        self.design_shape = design_shape
        self.resize_out = transforms.Resize(self.design_shape)

        # Spectral normalized projection
        self.proj = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512 * 7 * 7)),
            nn.ReLU(inplace=True),
        )

        # Spectral normalized deconvolutional layers
        self.deconv = nn.Sequential(
            # 7→13
            spectral_norm(
                nn.ConvTranspose2d(
                    512,
                    256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 13→25
            spectral_norm(
                nn.ConvTranspose2d(
                    256,
                    128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 25→50
            spectral_norm(
                nn.ConvTranspose2d(
                    128,
                    64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 50→100
            spectral_norm(
                nn.ConvTranspose2d(
                    64,
                    1,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        """Decode latent vector to image."""
        x = self.proj(z).view(z.size(0), 512, 7, 7)  # (B,512,7,7)
        x = self.deconv(x)  # (B,1,100,100)
        return self.resize_out(x)  # (B,1,H_orig,W_orig)


class LeastVolumeAE(nn.Module):
    """Barebones Least Volume Autoencoder.

    Combines reconstruction loss with volume loss to learn a compact latent representation.
    Volume loss = geometric mean of latent standard deviations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: th.optim.Optimizer,
        w_rec: float = 1.0,
        w_vol: float = 0.01,
        eta: float = 1e-4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.w_rec = w_rec
        self.w_vol = w_vol
        self.eta = eta

    def encode(self, x: th.Tensor) -> th.Tensor:
        """Encode design to latent representation."""
        return self.encoder(x)

    def decode(self, z: th.Tensor) -> th.Tensor:
        """Decode latent representation to design."""
        return self.decoder(z)

    def reconstruction_loss(self, x: th.Tensor, x_hat: th.Tensor) -> th.Tensor:
        """MSE reconstruction loss."""
        return th.nn.functional.mse_loss(x_hat, x)

    def volume_loss(self, z: th.Tensor) -> th.Tensor:
        """Volume loss = geometric mean of latent standard deviations.

        Encourages compact latent space by penalizing spread across dimensions.
        """
        std = z.std(dim=0)  # Standard deviation per latent dimension
        return th.exp(th.log(std + self.eta).mean())

    def loss(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute total loss = w_rec * reconstruction + w_vol * volume.

        Returns:
            total_loss, reconstruction_loss, volume_loss
        """
        z = self.encode(x)
        x_hat = self.decode(z)

        rec_loss = self.reconstruction_loss(x, x_hat)
        vol_loss = self.volume_loss(z)
        total_loss = self.w_rec * rec_loss + self.w_vol * vol_loss

        return total_loss, rec_loss, vol_loss

    def train_step(self, x: th.Tensor) -> tuple[float, float, float]:
        """Single training step.

        Returns:
            total_loss, reconstruction_loss, volume_loss (as Python floats)
        """
        self.optimizer.zero_grad()
        total_loss, rec_loss, vol_loss = self.loss(x)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), rec_loss.item(), vol_loss.item()


# ============================================================================
# Normalizing Flow Models
# ============================================================================


class CouplingLayer(nn.Module):
    """Affine coupling layer.

    Splits input in half, uses first half to compute scale and translation
    for second half. This maintains invertibility.
    """

    def __init__(self, dim: int, hidden_dim: int, n_conds: int = 0):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2

        # Network that computes scale and translation
        # Input: first half + conditions, Output: scale and translation for second half
        input_dim = self.split_dim + n_conds
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * 2),  # scale and translation
        )

    def forward(
        self, x: th.Tensor, c: th.Tensor | None = None, reverse: bool = False
    ) -> tuple[th.Tensor, th.Tensor]:
        """Forward or reverse pass through coupling layer.

        Args:
            x: Input (B, dim)
            c: Conditions (B, n_conds) or None
            reverse: If True, compute inverse transformation

        Returns:
            output, log_det_jacobian
        """
        # Split input
        x1, x2 = x[:, : self.split_dim], x[:, self.split_dim :]

        # Compute scale and translation from x1 (and conditions if provided)
        if c is not None:
            net_input = th.cat([x1, c], dim=1)
        else:
            net_input = x1

        net_out = self.net(net_input)
        log_s, t = net_out.chunk(2, dim=1)

        # Stabilize scale with tanh
        log_s = th.tanh(log_s)

        if not reverse:
            # Forward: x2' = x2 * exp(log_s) + t
            x2_out = x2 * th.exp(log_s) + t
            log_det = log_s.sum(dim=1)
        else:
            # Reverse: x2 = (x2' - t) / exp(log_s)
            x2_out = (x2 - t) * th.exp(-log_s)
            log_det = -log_s.sum(dim=1)

        return th.cat([x1, x2_out], dim=1), log_det


class NormalizingFlow(nn.Module):
    """Simple normalizing flow with coupling layers."""

    def __init__(self, dim: int, n_flows: int, hidden_dim: int, n_conds: int = 0):
        super().__init__()
        self.dim = dim
        self.n_conds = n_conds

        # Stack of coupling layers with alternating splits
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(CouplingLayer(dim, hidden_dim, n_conds))

        # Base distribution (standard Gaussian)
        self.register_buffer("base_loc", th.zeros(dim))
        self.register_buffer("base_scale", th.ones(dim))

    def forward(
        self, x: th.Tensor, c: th.Tensor | None = None
    ) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass: x -> z (design to latent).

        Returns:
            z, log_det_jacobian
        """
        log_det_total = th.zeros(x.shape[0], device=x.device)

        z = x
        for i, flow in enumerate(self.flows):
            z, log_det = flow(z, c, reverse=False)
            log_det_total += log_det

            # Permute for next layer (simple reversal)
            if i < len(self.flows) - 1:
                z = th.flip(z, dims=[1])

        return z, log_det_total

    def inverse(self, z: th.Tensor, c: th.Tensor | None = None) -> th.Tensor:
        """Inverse pass: z -> x (latent to design)."""
        x = z

        for i, flow in reversed(list(enumerate(self.flows))):
            # Undo permutation
            if i < len(self.flows) - 1:
                x = th.flip(x, dims=[1])

            x, _ = flow(x, c, reverse=True)

        return x

    def log_prob(self, x: th.Tensor, c: th.Tensor | None = None) -> th.Tensor:
        """Compute log probability of x under the model."""
        z, log_det = self.forward(x, c)

        # Log probability under base distribution (standard Gaussian)
        log_p_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)

        # Add log determinant of Jacobian
        log_p_x = log_p_z + log_det

        return log_p_x

    def sample(self, n_samples: int, c: th.Tensor | None = None) -> th.Tensor:
        """Sample from the model."""
        # Sample from base distribution
        z = th.randn(n_samples, self.dim, device=self.base_loc.device)

        # Transform through inverse flow
        x = self.inverse(z, c)

        return x


# ============================================================================
# Diffusion Models
# ============================================================================


class SimpleUNet(nn.Module):
    """Simple U-Net for denoising with time and condition embedding.

    Takes noisy image, timestep, and conditions as input.
    Outputs predicted noise.
    """

    def __init__(self, n_conds: int, time_emb_dim: int = 128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Condition embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_conds, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder (downsampling)
        self.enc1 = self._conv_block(1, 64)  # 100x100
        self.enc2 = self._conv_block(64, 128)  # 50x50
        self.enc3 = self._conv_block(128, 256)  # 25x25

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)  # 12x12

        # Decoder (upsampling) - learnable transposed convolutions
        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512 + 256, 256)  # concat with enc3
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256 + 128, 128)  # concat with enc2
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128 + 64, 64)  # concat with enc1

        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # Embedding projection to match channel dimensions
        self.emb_proj = nn.Linear(time_emb_dim * 2, 512)  # time + cond

        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Basic convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: th.Tensor, t: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Noisy image (B, 1, H, W)
            t: Timestep (B, 1)
            c: Conditions (B, n_conds)

        Returns:
            Predicted noise (B, 1, H, W)
        """
        # Embed time and conditions
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        c_emb = self.cond_mlp(c)  # (B, time_emb_dim)
        emb = th.cat([t_emb, c_emb], dim=1)  # (B, time_emb_dim * 2)
        emb = self.emb_proj(emb)  # (B, 512)

        # Encoder
        e1 = self.enc1(x)  # (B, 64, 100, 100)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 50, 50)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 25, 25)

        # Bottleneck with embedding
        b = self.bottleneck(self.pool(e3))  # (B, 512, 12, 12)
        # Add embedding as bias across spatial dimensions
        b = b + emb.view(-1, 512, 1, 1)

        # Decoder with skip connections and learned upsampling
        d3 = self.up3(b)  # (B, 512, 24, 24)
        # Crop or pad to match e3 size (25x25)
        d3 = nn.functional.interpolate(
            d3, size=e3.shape[2:], mode="bilinear", align_corners=True
        )
        d3 = self.dec3(th.cat([d3, e3], dim=1))  # (B, 256, 25, 25)

        d2 = self.up2(d3)  # (B, 256, 50, 50)
        d2 = self.dec2(th.cat([d2, e2], dim=1))  # (B, 128, 50, 50)

        d1 = self.up1(d2)  # (B, 128, 100, 100)
        d1 = self.dec1(th.cat([d1, e1], dim=1))  # (B, 64, 100, 100)

        return self.out(d1)  # (B, 1, 100, 100)


class DiffusionModel:
    """Simple diffusion model with linear noise schedule."""

    def __init__(
        self, num_timesteps: int, beta_start: float, beta_end: float, device: th.device
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear noise schedule
        self.betas = th.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=0)

        # Precompute values for sampling
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = th.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas
            * (
                1.0
                - th.cat([th.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
            )
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward diffusion: add noise to x_0 at timestep t.

        Returns:
            noisy_x, noise
        """
        noise = th.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )

        noisy_x = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x, noise

    @th.no_grad()
    def p_sample(
        self, model: nn.Module, x_t: th.Tensor, t: int, c: th.Tensor
    ) -> th.Tensor:
        """Reverse diffusion: denoise x_t at timestep t."""
        batch_size = x_t.shape[0]
        t_tensor = (
            th.full((batch_size, 1), t, device=self.device, dtype=th.float32)
            / self.num_timesteps
        )

        # Predict noise
        pred_noise = model(x_t, t_tensor, c)

        # Compute x_{t-1}
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Mean of p(x_{t-1} | x_t)
        x_t_minus_1_mean = (1.0 / th.sqrt(alpha)) * (
            x_t - (beta / th.sqrt(1.0 - alpha_cumprod)) * pred_noise
        )

        # Add noise if not final step
        if t > 0:
            noise = th.randn_like(x_t)
            variance = self.posterior_variance[t]
            x_t_minus_1 = x_t_minus_1_mean + th.sqrt(variance) * noise
        else:
            x_t_minus_1 = x_t_minus_1_mean

        return x_t_minus_1

    @th.no_grad()
    def sample(self, model: nn.Module, shape: tuple, c: th.Tensor) -> th.Tensor:
        """Sample from the model by running reverse diffusion."""
        batch_size = c.shape[0]
        # Start from pure noise
        x = th.randn(batch_size, *shape[1:], device=self.device)

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, c)

        return x


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # CGAN
    "CGANGenerator",
    "CGANDiscriminator",
    # LVAE
    "LVAE_Encoder",
    "LVAE_SNDecoder",
    # Flow
    "NormalizingFlow",
    "CouplingLayer",
    # Diffusion
    "SimpleUNet",
    "DiffusionModel",
]
