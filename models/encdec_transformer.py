import torch
import torch.nn as nn
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.resnet import Resnet1D


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self,
                 input_width=263,
                 output_width=402,
                 stride_t=2,
                 width=128,
                 nhead=16,
                 num_transformer_layers=1):
        super().__init__()

        # Convolution and ResNet blocks
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        # for i in range(down_t):
        #     input_dim = width
        #     block = nn.Sequential(
        #         nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
        #         nn.ReLU()
        #     )
        #     blocks.append(block)

        self.model = nn.Sequential(*blocks)
        self.pos_encoder = PositionalEncoding(width)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=width, nhead=nhead, dim_feedforward=width*2, dropout=0.1),
            num_layers=num_transformer_layers
        )

        self.up_model = nn.Sequential(nn.Conv1d(width, output_width, 3, 1, 1))

    def forward(self, x):
        x = self.model(x)  # Apply initial Conv and ResNet blocks
        x = x.permute(2, 0, 1)  # Change shape to (seq_len, batch, features) for Transformer
        x = self.pos_encoder(x)  # Apply positional encoding
        x = self.transformer_encoder(x)  # Apply Transformer
        x = x.permute(1, 2, 0)  # Revert shape back to (batch, features, seq_len)
        x = self.up_model(x)
        return x