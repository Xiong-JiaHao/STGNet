import torch
import torch.nn as nn
from model.gcn_lib import get_act_layer
from model.Self_Attention_Block import Attention_mask


class ROI_Selection_Block(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=[3, 3, 3], stride=1, padding=1, act="relu"):
        """
        Initialize the ROI_Selection_Block module.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            kernel_size (int or tuple): Kernel size for convolution. Default is [3, 3, 3].
            stride (int or tuple): Stride for convolution. Default is 1.
            padding (int or tuple): Padding size for convolution. Default is 1.
            act (str): Type of activation function. Default is "relu".

        """
        super(ROI_Selection_Block, self).__init__()
        self.data_conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(out_channel),
            get_act_layer(act),
        )
        self.attn_conv = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.Sigmoid(),
        )
        self.attn_mask = Attention_mask()
        self.out_conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
            nn.Tanh(),
        )

    def forward(self, inputs):  # B, C, T, H, W
        conv_out = self.data_conv(inputs)
        attent = self.attn_conv(conv_out)
        attent_mask = self.attn_mask(attent)
        out = conv_out * attent_mask
        out = self.out_conv(out)
        return out
