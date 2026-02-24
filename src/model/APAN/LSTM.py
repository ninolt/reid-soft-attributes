import torch
import torch.nn as nn

from src.config import CFG


class LSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = CFG["apan"]["lstm_kernel_size"], bias: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel size for same padding."
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def init_state(self, batch_size: int, spatial_size, device, dtype):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device, dtype=dtype)
        return h, c

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next