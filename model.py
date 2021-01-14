import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import tqdm

import layers
import sampler as sampler_module
import evaluation

class HeteroRGCN(nn.Module):
    def __init__(self, graph, target_node, node_feature, in_feats, h_dim, num_classes=2):
        super(HeteroRGCN, self).__init__()

        #       embed_dict = {ntype: nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device)) for ntype in graph.ntypes}
        embed_dict = {ntype: torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device) for ntype in graph.ntypes}

        for key, embed in embed_dict.items():
            embed_dict[key] = nn.init.zeros_(embed)
            # xavier_uniform_(embed) 如何可自动设置

        #    embed_dict['user'] = nn.Parameter(graph.nodes['user'].data['f'].float())

        self.embed = embed_dict
        self.embed[target_node] = node_feature

        self.layer1 = HeteroRGCNLayer(in_feats, h_dim, graph.etypes)
        self.layer2 = HeteroRGCNLayer(h_dim, num_classes, graph.etypes)
        # self.layer1 = RelGraphConv(in_feats, h_dim, num_rels=2)
        # self.layer2 = RelGraphConv(h_dim, num_classes, num_rels=2)

    def forward(self, graph):
        h_dict = self.layer1(graph, self.embed)
        h_dict_2 = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(graph, h_dict_2)
        # add softmax here
        return h_dict, h_dict_2  # ['user'] #改成通用的