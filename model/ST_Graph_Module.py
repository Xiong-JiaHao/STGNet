import torch.nn as nn
from model.ROI_Selection_Block import ROI_Selection_Block
from model.ST_Graph_Denoise_Block import ST_Graph_Denoise_Block


class ST_Graph_Moudle(nn.Module):

    def __init__(self, opt):
        """
        Initialize the Spatio-Temporal Graph Module.

        Args:
            opt (argparse.Namespace): Options for configuring the module.
        """
        super(ST_Graph_Moudle, self).__init__()
        channel = opt.stgm_in_channel  # Number of input channels
        act = opt.act_layer  # Activation function
        time_channel = opt.stgm_time_channel  # Number of time channels in each layer
        dropout_rate = opt.stgm_gnn_dropout_rate  # Dropout rate for each layer
        k = opt.stgm_gnn_k  # Number of neighbors for each layer
        norm = opt.norm  # Normalization method
        bias = opt.stgm_gnn_bias  # Whether to use bias
        epsilon = opt.stgm_gnn_epsilon  # Epsilon value for normalization
        stochastic = opt.stgm_gnn_stochastic  # Whether to use stochastic sampling
        gnn_conv = opt.stgm_gnn_conv  # Type of graph convolution
        pos_size = opt.stgm_pos_size  # Positional size in each layer
        gnn_num = opt.stgm_gnn_num  # Number of graph convolution layers in each layer

        self.st_graph = nn.ModuleList([])
        for id in range(len(time_channel) - 1):
            time_stride = int(time_channel[id] / time_channel[id + 1])
            pos_stride = int(pos_size[id] / pos_size[id + 1])
            stride = [time_stride, pos_stride, pos_stride]
            for _ in range(gnn_num[id]):
                self.st_graph += [ST_Graph_Denoise_Block(channel, time_channel[id], pos_size[id], k[id], act, norm,
                                                         bias, epsilon, stochastic, gnn_conv, dropout_rate[id])]
            self.st_graph += [ROI_Selection_Block(channel, channel, stride=stride, act=act)]

    def forward(self, inputs):  # [B, 64, 40, 8, 8]
        out = inputs
        for i in range(len(self.st_graph)):
            out = self.st_graph[i](out)

        # [B, 64, 20, 4, 4]
        return out