import torch
from torch import nn
from model.Stem_Module import Stem_Module
from model.ST_Graph_Module import ST_Graph_Moudle
from model.BVP_Prediction_Module import BVP_Prediction_Module


class STGNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.opt = OptInit()
        self.stem = Stem_Module(self.opt)
        self.STGM = ST_Graph_Moudle(self.opt)
        self.BPM = BVP_Prediction_Module(self.opt)

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x):  # [B, 3, 160, 64, 64]
        stem_out = self.stem(x)  # [B, 64, 40, 8, 8]
        stgm_out = self.STGM(stem_out)  # [B, 64, 20, 4, 4]
        wave = self.BPM(stgm_out)  # [B, 160]
        return wave

    def __str__(self):
        str = self.opt.__str__() + "\n"
        str = str + self.stem.__str__() + "\n"
        str = str + self.STGM.__str__() + "\n"
        str = str + self.BPM.__str__() + "\n"
        return str

    def save_model(self, log):
        torch.save(self.state_dict(), log + '/' + 'model_hr_best.pkl')


class OptInit:
    def __init__(self):
        # general
        self.act_layer = 'gelu'
        self.norm = 'batch'

        # Stem
        self.stem_in_channel = 3 * 2
        self.stem_channel = [16, 16, 16, 32, 32, 32, 64, 64]
        self.stem_downsample_status = [2, -1, 0, 1, -1, 0, 2, 0]

        # STGDM
        self.stgm_in_channel = 64
        self.stgm_time_channel = [40, 40, 20, 20]
        self.stgm_pos_size = [8, 8, 4, 4]
        self.stgm_gnn_num = [2, 3, 2, 1]
        self.stgm_gnn_dropout_rate = [0.25, 0.25, 0.25]
        self.stgm_gnn_k = [64, 32, 16, 16]
        self.stgm_gnn_conv = "avg_relative_conv"
        self.stgm_gnn_bias = True
        self.stgm_gnn_epsilon = 0.2
        self.stgm_gnn_stochastic = False

        # BPM
        self.bpm_in_channel = 64

    def __str__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())
