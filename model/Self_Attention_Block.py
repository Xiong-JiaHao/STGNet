import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Self_Attention_Block(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=[3, 3, 3], stride=1, padding=1, dropout_rate=0):
        super(Self_Attention_Block, self).__init__()
        self.attn_conv = nn.Conv3d(in_channel, in_channel, [1, 5, 5], stride=1, padding=[0, 2, 2])
        self.attn_mask = Attention_mask()
        self.drop = nn.Dropout(dropout_rate)
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)

    def forward(self, inputs):  # B, C, T, H, W
        attent = torch.sigmoid(self.attn_conv(inputs))
        attent_mask = self.attn_mask(attent)
        out = inputs * attent_mask
        out = self.drop(out)
        out = torch.tanh(self.conv(out))
        return out
