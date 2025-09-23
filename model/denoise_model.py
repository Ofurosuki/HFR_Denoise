from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils: circular padding on W
# -----------------------------
def circular_pad_w(x: torch.Tensor, pad_w: Tuple[int, int]) -> torch.Tensor:
    """
    Circular pad along W dimension (dim=3) for a 5D tensor (B, C, H, W, D).
    pad_w: (left, right)
    """
    if pad_w == (0, 0):
        return x
    left, right = pad_w
    # Take slices along W
    if left > 0:
        left_slice = x[:, :, :, -left:, :]
    else:
        left_slice = x.new_empty(x.shape[0], x.shape[1], x.shape[2], 0, x.shape[4])

    if right > 0:
        right_slice = x[:, :, :, :right, :]
    else:
        right_slice = x.new_empty(x.shape[0], x.shape[1], x.shape[2], 0, x.shape[4])

    x = torch.cat([left_slice, x, right_slice], dim=3)
    return x


def pad_h_d(x: torch.Tensor, pad_h: Tuple[int, int], pad_d: Tuple[int, int]) -> torch.Tensor:
    """
    Zero pad along H (dim=2) and D (dim=4), for a 5D tensor (B, C, H, W, D).
    """
    phl, phr = pad_h
    pdl, pdr = pad_d
    # F.pad expects pad for last dims reversed order: (D_right, D_left, W_right, W_left, H_right, H_left)
    return F.pad(x, (pdl, pdr, 0, 0, phl, phr), mode="constant", value=0.0)


# ------------------------------------------
# Conv3d with circular pad on W (wrapper)
# ------------------------------------------
class Conv3dCircularW(nn.Module):
    """
    3D Conv wrapper that applies circular padding on W, zero padding on H/D.
    Intended for stride=(1, sW, sD), kernel with odd sizes, typical padding='same'.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # manual padding
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # compute "same" padding for H/W/D
        kh, kw, kd = kernel_size
        dh, dw, dd = dilation
        # effective kernel = (k-1)*d + 1
        eff_kh = (kh - 1) * dh + 1
        eff_kw = (kw - 1) * dw + 1
        eff_kd = (kd - 1) * dd + 1

        self.pad_h = (eff_kh // 2, eff_kh // 2)
        self.pad_w = (eff_kw // 2, eff_kw // 2)
        self.pad_d = (eff_kd // 2, eff_kd // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D)
        x = circular_pad_w(x, self.pad_w)
        x = pad_h_d(x, self.pad_h, self.pad_d)
        x = self.conv(x)
        return x


# -----------------------------------------------------
# Depthwise-Separable 3D Conv block (DW3D + PW1x1x1)
# with GroupNorm + GELU; W uses circular padding
# -----------------------------------------------------
class DWSeparable3DBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (3, 5, 7),
        stride: Tuple[int, int, int] = (1, 1, 1),
        groups_gn: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()
        # depthwise
        self.dw = Conv3dCircularW(
            in_ch, in_ch,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_ch,
            bias=False
        )
        # pointwise
        self.pw = Conv3dCircularW(
            in_ch, out_ch,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=False
        )
        self.norm = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.use_residual = use_residual and (in_ch == out_ch) and (stride == (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        if self.use_residual:
            x = x + identity
        return x


# ------------------------------------------
# Downsample block (anisotropic, circular W)
# ------------------------------------------
class Downsample3D(nn.Module):
    def __init__(self, ch: int, stride=(1, 2, 2)):
        super().__init__()
        self.conv = Conv3dCircularW(
            ch, ch,
            kernel_size=(3, 3, 3),
            stride=stride,
            bias=False
        )
        self.norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ---------------------------------------------------
# Axial Self-Attention along W or D axes
# ---------------------------------------------------
class AxialSelfAttentionW(nn.Module):
    def __init__(self, ch: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.mha = nn.MultiheadAttention(ch, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D)
        B, C, H, W, D = x.shape
        y = self.norm(x)
        # merge (B, H, D) as batch; sequence = W; feature = C
        y = y.permute(0, 2, 4, 3, 1).contiguous().view(B * H * D, W, C)  # (N, S = W, C)
        out, _ = self.mha(y, y, y, need_weights=False)
        out = out.view(B, H, D, W, C).permute(0, 4, 1, 3, 2).contiguous()  # (B, C, H, W, D)
        return x + out  # residual


class AxialSelfAttentionD(nn.Module):
    def __init__(self, ch: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.mha = nn.MultiheadAttention(ch, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D)
        B, C, H, W, D = x.shape
        y = self.norm(x)
        # merge (B, H, W) as batch; sequence = D; feature = C
        y = y.permute(0, 2, 3, 4, 1).contiguous().view(B * H * W, D, C)  # (N, S = D, C)
        out, _ = self.mha(y, y, y, need_weights=False)
        out = out.view(B, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, H, W, D)
        return x + out  # residual


# -----------------------------
# Model for HFR Attack Denoising
# -----------------------------
class DenoiseModel(nn.Module):
    """
    Input: (B, H, W, D)
    Output: (B, 3, H, W, D)
    hidden_dim: Base hidden dimension
    use_axial_attn: use W/D axial attention
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        hidden_dim: int = 32,
        use_axial_attn: bool = False,
    ):
        super().__init__()
        C = hidden_dim

        # Stem: Feature extraction and downsample
        self.stem = nn.Sequential(
            Conv3dCircularW(in_channels, C, kernel_size=(3, 7, 9), stride=(1, 5, 8), bias=False),
            nn.GroupNorm(num_groups=min(8, C), num_channels=C),
            nn.GELU(),
        )

        # Encoder Stages (DW+PW blocks)
        self.enc0 = nn.Sequential(
            DWSeparable3DBlock(C, C, kernel_size=(3, 5, 7)),
            DWSeparable3DBlock(C, C, kernel_size=(3, 5, 7)),
        )  # keep (H, 360, 100)

        self.down1 = Downsample3D(C, stride=(1, 2, 2))  # -> (H, 180, 50)
        self.enc1 = nn.Sequential(
            DWSeparable3DBlock(C, 2 * C, kernel_size=(3, 5, 7)),
            DWSeparable3DBlock(2 * C, 2 * C, kernel_size=(3, 3, 5)),
        )

        self.down2 = Downsample3D(2 * C, stride=(1, 2, 2))  # -> (H, 90, 25)
        self.enc2 = nn.Sequential(
            DWSeparable3DBlock(2 * C, 4 * C, kernel_size=(3, 3, 5)),
            DWSeparable3DBlock(4 * C, 4 * C, kernel_size=(3, 3, 5)),
        )

        self.down3 = Downsample3D(4 * C, stride=(1, 2, 1))  # -> (H, 45, 25)
        self.bottleneck = nn.Sequential(
            DWSeparable3DBlock(4 * C, 8 * C, kernel_size=(3, 3, 5)),
            DWSeparable3DBlock(8 * C, 8 * C, kernel_size=(3, 3, 5)),
        )

        self.use_axial_attn = use_axial_attn
        if use_axial_attn:
            self.ax_w = AxialSelfAttentionW(8 * C, num_heads=8)
            self.ax_d = AxialSelfAttentionD(8 * C, num_heads=8)

        # Decoder (upsample + concat skip + light blocks)
        self.up3 = nn.Upsample(scale_factor=(1, 2, 1), mode="trilinear", align_corners=False)
        self.dec3 = nn.Sequential(
            Conv3dCircularW(8 * C + 4 * C, 4 * C, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.GroupNorm(num_groups=min(8, 4 * C), num_channels=4 * C),
            nn.GELU(),
            DWSeparable3DBlock(4 * C, 4 * C, kernel_size=(3, 3, 5)),
        )

        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            Conv3dCircularW(4 * C + 2 * C, 2 * C, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.GroupNorm(num_groups=min(8, 2 * C), num_channels=2 * C),
            nn.GELU(),
            DWSeparable3DBlock(2 * C, 2 * C, kernel_size=(3, 3, 5)),
        )

        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            Conv3dCircularW(2 * C + C, C, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.GroupNorm(num_groups=min(8, C), num_channels=C),
            nn.GELU(),
            DWSeparable3DBlock(C, C, kernel_size=(3, 5, 7)),
        )

        # Head up to full resolution: (1,5,8) back to (H, 1800, 800)
        self.head_up = nn.Upsample(scale_factor=(1, 5, 8), mode="trilinear", align_corners=False)
        self.head = Conv3dCircularW(C, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)

    def forward(self, x):
        """
        Input: (B, H, W, D)
        Returns: (B, num_classes, H, W, D)
        """
        x = x.unsqueeze(1)  # (B, 1, H, W, D)

        # Stem
        x0 = self.stem(x)  # (B, C, H, 360, 100)

        # Encoder
        e0 = self.enc0(x0)  # (B, C, H, 360, 100)
        d1 = self.down1(e0)  # (B, C, H, 180, 50)

        e1 = self.enc1(d1)  # (B, 2C, H, 180, 50)
        d2 = self.down2(e1)  # (B, 2C, H, 90, 25)

        e2 = self.enc2(d2)  # (B, 4C, H, 90, 25)
        d3 = self.down3(e2)  # (B, 4C, H, 45, 25)

        b = self.bottleneck(d3)  # (B, 8C, H, 45, 25)

        if self.use_axial_attn:
            b = self.ax_w(b)
            b = self.ax_d(b)

        # Decoder
        u3 = self.up3(b)  # (B, 8C, H, 90, 25)
        u3 = torch.cat([u3, e2], dim=1)  # (B, 8C+4C, H, 90, 25)
        u3 = self.dec3(u3)  # (B, 4C, H, 90, 25)

        u2 = self.up2(u3)  # (B, 4C, H, 180, 50)
        u2 = torch.cat([u2, e1], dim=1)  # (B, 4C+2C, H, 180, 50)
        u2 = self.dec2(u2)  # (B, 2C, H, 180, 50)

        u1 = self.up1(u2)  # (B, 2C, H, 360, 100)
        u1 = torch.cat([u1, e0], dim=1)  # (B, 2C+C, H, 360, 100)
        u1 = self.dec1(u1)  # (B, C, H, 360, 100)

        # Head to full resolution
        h = self.head_up(u1)  # (B, C, H, 1800, 800)
        logits = self.head(h)  # (B, num_classes, H, 1800, 800)

        return logits


# -----------------------------
# Test Bench
# -----------------------------
if __name__ == "__main__":
    B, H, W, D = 4, 32, 1800, 800
    x = torch.randn(B, H, W, D).to('cuda')
    model = DenoiseModel(in_channels=1, num_classes=3, hidden_dim=32, use_axial_attn=False).to('cuda')
    with torch.no_grad():
        y = model(x)
    print(y)
