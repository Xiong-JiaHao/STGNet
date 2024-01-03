import torch.nn as nn
from model.Self_Attention_Block import Self_Attention_Block
from model.Stem_Block import Stem_Block
from model.Diff_Block import Diff_Block


class Stem_Module(nn.Module):

    def __init__(self, opt):
        super(Stem_Module, self).__init__()
        channel = opt.stem_in_channel
        stem_channel = opt.stem_channel
        downsample_status = opt.stem_downsample_status

        act = opt.act_layer

        self.diff_block = Diff_Block()
        self.stem = nn.ModuleList([])
        for ids, item in enumerate(stem_channel):
            if downsample_status[ids] == 2:
                self.stem += [Stem_Block(channel, item, stride=2, act=act)]
            elif downsample_status[ids] == 1:
                self.stem += [Stem_Block(channel, item, stride=[1, 2, 2], act=act)]
            elif downsample_status[ids] == -1:
                self.stem += [Self_Attention_Block(channel, item)]
            else:
                self.stem += [Stem_Block(channel, item, act=act)]
            channel = item

    def forward(self, inputs):  # [B, 3, 160, 64, 64]
        out = self.diff_block(inputs)  # [B, 6, 160, 64, 64]
        for i in range(len(self.stem)):
            out = self.stem[i](out)
        return out  # [B, 64, 40, 8, 8]
