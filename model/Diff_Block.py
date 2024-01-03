import torch
import torch.nn as nn


class Diff_Block(torch.nn.Module):

    def __init__(self):
        super(Diff_Block, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1, 3, 4)  # [B, 160, 3, 64, 64]
        B, T, C, H, W = x.shape
        # 添加多一帧, for diff
        last_frame = x[:, -1, :, :, :].unsqueeze(1)
        inputs = torch.cat([x, last_frame], 1)  # [B, 160+1, 3, 64, 64]
        inputs = torch.diff(inputs, dim=1)  # [B, 160, 3, 64, 64]
        inputs = inputs.reshape(B * T, C, H, W)
        inputs = self.batch_norm(inputs)
        out = inputs.reshape(B, T, C, H, W)  # [B, 160, 3, 64, 64]
        out = torch.cat([x, out], dim=2)  # [B, 160, 6, 64, 64]

        output = out.permute(0, 2, 1, 3, 4)  # [B, 6, 160, 64, 64]
        return output
