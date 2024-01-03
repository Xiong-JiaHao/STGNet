import torch.nn as nn
from data import ClipFramesLen


class BVP_Prediction_Module(nn.Module):

    def __init__(self, opt):
        super(BVP_Prediction_Module, self).__init__()
        channel = opt.bpm_in_channel
        act = opt.act_layer
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(channel, channel, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(channel),
            nn.ELU(),
            nn.ConvTranspose3d(channel, channel, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(channel),
            nn.ELU(),
            nn.ConvTranspose3d(channel, channel, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(channel),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((ClipFramesLen, 1, 1)),
            nn.Conv3d(channel, 1, [1, 1, 1], stride=1, padding=[0, 0, 0]),
        )

    def forward(self, inputs):  # [B, 64, 20, 4, 4]
        B, _, _, _, _ = inputs.shape
        out = self.upsample(inputs)
        out = out.reshape(B, ClipFramesLen)
        return out