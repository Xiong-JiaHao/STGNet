import torch
from torch import nn
from model.Stem_Module import Stem_Module
from model.ST_Graph_Module import ST_Graph_Moudle
from model.BVP_Prediction_Module import BVP_Prediction_Module


class STGNet(nn.Module):

    def __init__(self):
        """
        Initialize the Spatio-Temporal Graph Network.

        The network consists of a stem module, a spatio-temporal graph module, and a blood volume pulse prediction module.
        """
        super().__init__()
        self.opt = OptInit()  # Initialize options
        self.stem = Stem_Module(self.opt)  # Initialize the stem module
        self.STGM = ST_Graph_Moudle(self.opt)  # Initialize the spatio-temporal graph module
        self.BPM = BVP_Prediction_Module(self.opt)  # Initialize the blood volume pulse prediction module

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the weights of the network.
        """
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
        """
        Return a string representation of the network configuration.
        """
        str = self.opt.__str__() + "\n"
        str = str + self.stem.__str__() + "\n"
        str = str + self.STGM.__str__() + "\n"
        str = str + self.BPM.__str__() + "\n"
        return str

    def save_model(self, log):
        """
        Save the model's weights to a file.

        Args:
            log (str): Path to the directory where the model weights will be saved.
        """
        torch.save(self.state_dict(), log + '/' + 'model_hr_best.pkl')


class OptInit:
    def __init__(self):
        """
        Initialize the options for configuring the network.
        """
        # General options
        self.act_layer = 'gelu'  # Activation function
        self.norm = 'batch'  # Normalization method

        # Stem options
        self.stem_in_channel = 3 * 2  # Number of input channels for the stem module
        self.stem_channel = [16, 16, 16, 32, 32, 32, 64, 64]  # Channels in each layer of the stem module
        self.stem_downsample_status = [2, -1, 0, 1, -1, 0, 2, 0]  # Downsample status for each layer

        # Spatio-temporal graph module (STGM) options
        self.stgm_in_channel = 64  # Number of input channels for the STGM
        self.stgm_time_channel = [40, 40, 20, 20]  # Number of time channels in each layer of the STGM
        self.stgm_pos_size = [8, 8, 4, 4]  # Size in each layer of the STGM
        self.stgm_gnn_num = [2, 3, 2, 1]  # Number of graph convolution layers in each layer of the STGM
        self.stgm_gnn_dropout_rate = [0.25, 0.25, 0.25]  # Dropout rate for each layer of the STGM
        self.stgm_gnn_k = [64, 32, 16, 16]  # Number of neighbors for each layer of the STGM
        self.stgm_gnn_conv = "avg_relative_conv"  # Type of graph convolution in the STGM
        self.stgm_gnn_bias = True  # Whether to use bias in graph convolution
        self.stgm_gnn_epsilon = 0.2  # Epsilon value for normalization in graph convolution
        self.stgm_gnn_stochastic = False  # Whether to use stochastic sampling in graph convolution

        # Blood volume pulse prediction module (BPM) options
        self.bpm_in_channel = 64  # Number of input channels for the BPM

    def __str__(self):
        """
        Return a string representation of the options.
        """
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())
