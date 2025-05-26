import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_transformer import DiTConVBlock
from utils.mask import sequence_mask

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
        
class SFM(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, output_channels):
        super().__init__()
        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, output_channels, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


# modified from https://github.com/jaywalnut310/vits/blob/main/models.py
class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.scale = self.hidden_channels ** 0.5

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = nn.ModuleList([DiTConVBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout, gin_channels) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

        self.sfm = SFM(hidden_channels, filter_channels, kernel_size, p_dropout, out_channels+2)
        self.proj_sfm = nn.Conv1d(hidden_channels, out_channels, 1)

        self.prenet = nn.Sequential(
            nn.Conv1d(hidden_channels, filter_channels, kernel_size, padding=kernel_size//2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2), # add about 3M params
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for block in self.encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, x_lengths: torch.Tensor):
        x = self.emb(x) * self.scale  # [b, t, h]
        x = x.transpose(1, -1)  # [b, h, t]
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        for layer in self.encoder:
            x = layer(x, c, x_mask)
        mu_x = self.proj(x) * x_mask

        return x, mu_x, x_mask

    def forward_smooth(self, x, c, x_mask):
        x = self.prenet(x * x_mask)
        mu = self.proj_sfm(x) * x_mask

        x = self.sfm(x, x_mask)
        tp = x[:, :1, :]
        logsigma_p = x[:, 1:2, :]
        xp = x[:, 2:, :]
        tp = F.sigmoid(tp) * x_mask
        tp = torch.sum(tp, dim=-1, keepdim=True) / torch.sum(x_mask, dim=-1, keepdim=True) #[b, 1, 1]
        logsigma_p = torch.sum(logsigma_p, dim=-1, keepdim=True) / torch.sum(x_mask, dim=-1, keepdim=True) #[b, 1, 1]

        return mu, [xp, tp, logsigma_p]