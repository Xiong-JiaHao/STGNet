import torch
import torch.nn as nn
from model.gcn_lib import get_act_layer


class Stem_Block(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=[3, 3, 3], stride=1, padding=1, act="relu"):
        super(Stem_Block, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            get_act_layer(act),
        )

    def forward(self, inputs):  # B, C, T, H, W
        out = self.stem(inputs)
        return out