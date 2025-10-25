from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1D Depthwise-Separable Block
# -----------------------------
class DWSeparable1DBlock(nn.Module):
    """
    1D Separable Depthwise Conv
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 7,
        stride: int = 1,
        groups_gn: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()
        pad = k // 2
        # depthwise: groups=in_ch
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride,
                            padding=pad, groups=in_ch, bias=False)
        # pointwise
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.use_residual = (use_residual and stride == 1 and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        if self.use_residual:
            x = x + identity
        return x


# -----------------------------
# Downsample / Upsample along D
# -----------------------------
class Downsample1D(nn.Module):
    def __init__(self, ch: int, stride: int = 2, k: int = 3):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpSample1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int):
        super().__init__()
        self.scale = scale
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale != 1:
            L = x.shape[-1]
            x = F.interpolate(x, size=L * self.scale, mode="linear", align_corners=False)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# -----------------------------
# D Axis Attention
# -----------------------------
class DAxisAttention(nn.Module):
    def __init__(self, ch: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.mha = nn.MultiheadAttention(ch, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        y = self.norm(x)
        y = y.transpose(1, 2)  # (N, L, C)
        out, _ = self.mha(y, y, y, need_weights=False)
        out = out.transpose(1, 2)  # (N, C, L)
        return x + out


# -----------------------------
# Baseline: D-only 1D Conv UNet
# -----------------------------
class DAxisConvBaseline(nn.Module):
    """
    Baseline model: 1D CONV on D axis
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        hidden_dim: int = 32,
        use_d_attn: bool = True,
    ):
        super().__init__()
        C = hidden_dim

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=9, stride=8, padding=4, bias=False),  # D: 800 -> 100
            nn.GroupNorm(num_groups=min(8, C), num_channels=C),
            nn.GELU(),
        )

        # Encoder stages
        self.enc0 = nn.Sequential(
            DWSeparable1DBlock(C, C, k=7),
            DWSeparable1DBlock(C, C, k=7),
        )  # L: 100

        self.down1 = Downsample1D(C, stride=2, k=3)  # 100 -> 50
        self.enc1 = nn.Sequential(
            DWSeparable1DBlock(C, 2 * C, k=5),
            DWSeparable1DBlock(2 * C, 2 * C, k=5),
        )  # L: 50, C: 2C

        self.down2 = Downsample1D(2 * C, stride=2, k=3)  # 50 -> 25
        self.enc2 = nn.Sequential(
            DWSeparable1DBlock(2 * C, 4 * C, k=5),
            DWSeparable1DBlock(4 * C, 4 * C, k=5),
        )  # L: 25, C: 4C

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DWSeparable1DBlock(4 * C, 8 * C, k=3),
            DWSeparable1DBlock(8 * C, 8 * C, k=3),
        )
        self.use_d_attn = use_d_attn
        if use_d_attn:
            self.d_attn = DAxisAttention(8 * C, num_heads=8)

        # Decoder
        self.up2 = UpSample1D(8 * C, 4 * C, scale=1)  # 25 -> 25
        self.dec2 = nn.Sequential(
            nn.Conv1d(4 * C + 4 * C, 4 * C, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, 4 * C), num_channels=4 * C),
            nn.GELU(),
            DWSeparable1DBlock(4 * C, 4 * C, k=5),
        )

        self.up1 = UpSample1D(4 * C, 2 * C, scale=2)  # 25 -> 50
        self.dec1 = nn.Sequential(
            nn.Conv1d(2 * C + 2 * C, 2 * C, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, 2 * C), num_channels=2 * C),
            nn.GELU(),
            DWSeparable1DBlock(2 * C, 2 * C, k=5),
        )

        self.up0 = UpSample1D(2 * C, C, scale=2)  # 50 -> 100
        self.dec0 = nn.Sequential(
            nn.Conv1d(C + C, C, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, C), num_channels=C),
            nn.GELU(),
            DWSeparable1DBlock(C, C, k=7),
        )

        # Head
        self.head_up = UpSample1D(C, C, scale=8)  # 100 -> 800
        self.head = nn.Conv1d(C, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, D)
        return: (B, num_classes, H, W, D)
        """
        B, H, W, D = x.shape
        N = B * H * W
        # Flatten spatial data to 1D sequence
        x = x.reshape(N, 1, D)  # (N, 1, D)

        # Stem & Encoder
        x0 = self.stem(x)    # (N, C, 100)
        e0 = self.enc0(x0)   # (N, C, 100)

        d1 = self.down1(e0)  # (N, C, 50)
        e1 = self.enc1(d1)   # (N, 2C, 50)

        d2 = self.down2(e1)  # (N, 2C, 25)
        e2 = self.enc2(d2)   # (N, 4C, 25)

        # Bottleneck
        b = self.bottleneck(e2)  # (N, 8C, 25)
        if self.use_d_attn:
            b = self.d_attn(b)   # (N, 8C, 25)

        # Decoder with skip
        u2 = self.up2(b)                                # (N, 4C, 25)
        u2 = torch.cat([u2, e2], dim=1)                 # (N, 8C, 25)
        u2 = self.dec2(u2)                              # (N, 4C, 25)

        u1 = self.up1(u2)                               # (N, 2C, 50)
        u1 = torch.cat([u1, e1], dim=1)                 # (N, 4C, 50)
        u1 = self.dec1(u1)                              # (N, 2C, 50)

        u0 = self.up0(u1)                               # (N, C, 100)
        u0 = torch.cat([u0, e0], dim=1)                 # (N, 2C, 100)
        u0 = self.dec0(u0)                              # (N, C, 100)

        # Head to full D
        h = self.head_up(u0)                            # (N, C, 800)
        logits = self.head(h)                           # (N, num_classes, 800)

        # (B, num_classes, H, W, D)
        logits = logits.view(B, H, W, -1, logits.shape[1])  # (B, H, W, D, C)
        logits = logits.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, H, W, D)
        return logits


# -----------------------------
# Test Bench
# -----------------------------
if __name__ == "__main__":
    import time
    B, H, W, D = 1, 32, 1800, 800
    x = torch.randn(B, H, W, D).cuda()
    model = DAxisConvBaseline(in_channels=1, num_classes=3, hidden_dim=32, use_d_attn=True).cuda()
    with torch.no_grad():
        st = time.time()
        y = model(x)
        ed = time.time()
        print('Frame process time: {}'.format(ed - st))
    print("input:", x.shape, "output:", y.shape)
