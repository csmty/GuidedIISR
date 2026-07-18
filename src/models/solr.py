import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, growth_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, num_layers=3, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_channels))
            channels += growth_channels
        self.layers = nn.Sequential(*layers)
        self.fuse = nn.Conv2d(channels, in_channels, 1, 1, 0)

    def forward(self, x):
        return x + self.residual_scale * self.fuse(self.layers(x))


class SOLR(nn.Module):
    """Second-Order Latent Residual Refinement for one-step latent denoising."""

    def __init__(
        self,
        latent_channels=4,
        hidden_channels=64,
        growth_channels=32,
        num_blocks=3,
        num_layers_per_block=3,
        residual_scale=0.2,
        kappa=0.2,
    ):
        super().__init__()
        self.kappa = kappa
        self.proj_z = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.proj_r = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        body_channels = hidden_channels * 4
        self.body = nn.Sequential(
            *[
                DenseBlock(
                    in_channels=body_channels,
                    growth_channels=growth_channels,
                    num_layers=num_layers_per_block,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out = nn.Conv2d(body_channels, latent_channels, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_l, r1):
        """Predict a bounded second-order residual from LR latent and first residual."""
        if z_l.shape != r1.shape:
            raise ValueError(
                f"SOLR expects z_l and r1 to have the same shape, "
                f"but got z_l={tuple(z_l.shape)}, r1={tuple(r1.shape)}"
            )
        fz = self.proj_z(z_l)
        fr = self.proj_r(r1)
        phi = torch.cat([fz, fr, fr * fr, fz * fr], dim=1)
        raw = self.out(self.body(phi))
        return self.kappa * torch.tanh(raw)
