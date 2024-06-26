import torch
import torch.nn as nn

from model.gcn_lib import get_act_layer
from timm.models.layers import DropPath
from model.gcn_lib.torch_edge import DenseDilatedKnnGraph
from model.gcn_lib.torch_vertex import GraphConv2d


class ST_Graph_Denoise_Block(torch.nn.Module):

    def __init__(self, channel, time_channel, pos_size=14, k=4, act='relu', norm=None, bias=True, epsilon=0.0,
                 stochastic=False, gnn_conv='edge', drop_path=0.0):
        """
        Initialize the Spatio-Temporal Graph Denoise Block.

        Args:
            channel (int): Number of input channels.
            time_channel (int): Size of the time dimension.
            pos_size (int): Size of the positional embedding.
            k (int): Number of nearest neighbors for graph convolution.
            act (str): Type of activation function. Default is 'relu'.
            norm (str or None): Type of normalization. Default is None.
            bias (bool): Whether to use bias in convolutional layers. Default is True.
            epsilon (float): Epsilon value for stability. Default is 0.0.
            stochastic (bool): Whether to use stochastic graph convolution. Default is False.
            gnn_conv (str): Type of graph convolution. Default is 'edge'.
            drop_path (float): Drop path rate. Default is 0.0.
        """
        super(ST_Graph_Denoise_Block, self).__init__()
        self.pos_size = pos_size
        self.pos_embed = nn.Parameter(torch.zeros(1, time_channel * pos_size * pos_size, channel))

        self.graph = SignalGraphConv2d(channel, channel * 2, k, gnn_conv, act, norm, bias,
                                       stochastic, epsilon)
        self.fc_1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            get_act_layer(act),
        )

        self.fc_2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            get_act_layer(act)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):  # B, C, T, H, W
        B, C, T, H, W = inputs.shape
        x = inputs.reshape(B, C, T * H * W)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.graph(x)
        x = self.fc_1(x)
        shortcut = x
        x = self.fc_2(x)
        x = self.drop_path(x) + shortcut
        output = x.reshape(B, C, T, H, W)
        return output


class SignalGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, dilation=1):
        """
        Initialize the SignalGraphConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            conv (str): Type of graph convolution. Default is 'edge'.
            act (str): Type of activation function. Default is 'relu'.
            norm (str or None): Type of normalization. Default is None.
            bias (bool): Whether to use bias in convolutional layers. Default is True.
            stochastic (bool): Whether to use stochastic graph convolution. Default is False.
            epsilon (float): Epsilon value for stability. Default is 0.0.
            dilation (int): Dilation rate for dilated graph convolution. Default is 1.
        """
        super(SignalGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None, **kwargs):
        y = None
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(SignalGraphConv2d, self).forward(x, edge_index, y)
        return x
