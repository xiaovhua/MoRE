import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GSC(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, reduction=4):
        super().__init__()
        
        out_channels = out_channels or in_channels
        mid_channels = out_channels // reduction

        self.proj1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 1, 1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, in_channels, 3, 1, 1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True)
        )

        self.proj2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, 1, 0),
            nn.GroupNorm(8, in_channels),
            nn.ReLU()
        )

        self.proj3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, 1, 0),
            nn.GroupNorm(8, in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x_residual = x 
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        x = x1 + x2
        x = self.proj3(x)
        return x + x_residual


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or 2 * hidden_size
        self.norm = nn.GroupNorm(8, hidden_size)
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x_residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + x_residual


class TSMamba(nn.Module):
    def __init__(self, dim, num_slices=4):
        super().__init__()
        self.gsc = GSC(dim)
        self.mamba = MambaLayer(dim, num_slices=num_slices)
        self.mlp = MlpChannel(dim)

        # print('*' * 100)
        # print(f"Params GSC: {sum([p.numel() for p in self.gsc.parameters()])}")
        # print(f"Params Mamba: {sum([p.numel() for p in self.mamba.parameters()])}")
        # print(f"Params MLP: {sum([p.numel() for p in self.mlp.parameters()])}")
        # print('*' * 100)

    def forward(self, x):
        return self.mlp(self.mamba(self.gsc(x)))


if __name__ == "__main__":
    x = torch.rand((1, 128, 32, 32, 24)).cuda()
    m = TSMamba(128).cuda()
    print(m(x).shape)
    print(f'Params: {sum([p.numel() for p in m.parameters()])}')

