"""
CyberShield PyTorch Models Architecture Definitions
Defines exact architectures for:
1. ConditionalGenerator (WGAN-GP Generator)
2. ConditionalCritic (WGAN-GP Critic)
3. TabularAutoencoder (Unsupervised Anomaly Detector)
4. CyberTransformer (Multiclass Intrusion Classifier)
5. CondUNet (Diffusion Noise Predictor for Traffic Heatmaps)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

class ConditionalGenerator(nn.Module):
    """
    Class-conditional WGAN-GP generator for tabular network flows.
    Input:
        z: Latent Gaussian noise vector of dimension latent_dim (128)
        labels: LongTensor of class labels (0 to num_classes-1)
    Output:
        x_gen: Tabular flow features (77) in gan_scaler space with Tanh activation [-1, 1].
    """
    def __init__(self, latent_dim: int = 128, num_classes: int = 11, embed_dim: int = 16, output_dim: int = 77):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class ConditionalCritic(nn.Module):
    """
    Class-conditional Critic (Wasserstein discriminator) with gradient penalty compatibility.
    No batch normalization is used in accordance with WGAN-GP principles.
    """
    def __init__(self, input_dim: int = 77, num_classes: int = 11, embed_dim: int = 16):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embed(labels)
        inp = torch.cat([x, c], dim=1)
        return self.net(inp)


class TabularAutoencoder(nn.Module):
    """
    Symmetric Tabular Autoencoder for baseline network flow reconstruction and anomaly detection.
    Compresses 77 features -> 128 -> 64 -> 32 -> 16 bottleneck and reconstructs back to 77.
    """
    def __init__(self, input_dim: int = 77, bottleneck: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.Linear(32, bottleneck),                      nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),                      nn.LeakyReLU(0.2),
            nn.Linear(32, 64),         nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class CyberTransformer(nn.Module):
    """
    Self-Attention Multi-head Transformer Encoder for tabular intrusion sequence and feature classification.
    Embeds each of the 77 flow features into d_model=64 space with positional encodings,
    passes through 3 TransformerEncoder layers, and classifies into 12 classes (BENIGN + 11 attacks).
    """
    def __init__(self, input_dim: int = 77, num_classes: int = 12,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, input_dim, d_model) * 0.1
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Reshape (B, 77) -> (B, 77, 1)
        x_seq = x.unsqueeze(-1)
        embedded = self.feature_embedding(x_seq) + self.pos_embedding
        transformed = self.transformer(embedded)
        flattened = transformed.reshape(batch_size, -1)
        return self.classifier(flattened)


# ── Diffusion Generative Model (U-Net with Text Conditioning) ───────────────

def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    """Residual block with group norm, SiLU, and time/text embedding projection."""
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.c1(Fn.silu(self.n1(x)))
        h = h + self.e(e)[:, :, None, None]
        h = self.c2(Fn.silu(self.n2(h)))
        return h + self.skip(x)


class SelfAttn(nn.Module):
    """Spatial self-attention block for 2D traffic heatmaps."""
    def __init__(self, c: int):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.o = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).unbind(1)
        h = Fn.scaled_dot_product_attention(
            q.transpose(1, 2)[:, None], k.transpose(1, 2)[:, None], v.transpose(1, 2)[:, None]
        )[:, 0]
        return x + self.o(h.transpose(1, 2).reshape(B, C, H, W))


class CondUNet(nn.Module):
    """
    Conditional U-Net that predicts noise in a 64x64 traffic heatmap.
    Conditioned on diffusion timestep + CLIP/text embedding.
    64x64 -> 32x32 -> 16x16 (attention) -> back up with skip connections.
    """
    def __init__(self, txt_dim: int = 512, base: int = 64, emb: int = 256):
        super().__init__()
        self.emb = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_mlp = nn.Sequential(nn.Linear(txt_dim, emb), nn.SiLU(), nn.Linear(emb, emb))

        self.inc = nn.Conv2d(1, base, 3, padding=1)
        self.d1 = ResBlock(base, base, emb)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.d2 = ResBlock(base * 2, base * 2, emb)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.d3 = ResBlock(base * 4, base * 4, emb)
        self.a3 = SelfAttn(base * 4)

        self.mid1 = ResBlock(base * 4, base * 4, emb)
        self.mida = SelfAttn(base * 4)
        self.mid2 = ResBlock(base * 4, base * 4, emb)

        self.u3 = ResBlock(base * 8, base * 4, emb)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.u2 = ResBlock(base * 4, base * 2, emb)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.u1 = ResBlock(base * 2, base, emb)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 1, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        e = self.t_mlp(sinusoidal(t, self.emb)) + self.c_mlp(c)
        h0 = self.d1(self.inc(x), e)
        h1 = self.d2(self.down1(h0), e)
        h2 = self.a3(self.d3(self.down2(h1), e))
        m = self.mid2(self.mida(self.mid1(h2, e)), e)
        u = self.u3(torch.cat([m, h2], 1), e)
        u = self.u2(torch.cat([self.up2(u), h1], 1), e)
        u = self.u1(torch.cat([self.up1(u), h0], 1), e)
        return self.out(u)

