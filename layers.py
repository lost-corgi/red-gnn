import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, graph, feat_dict):
        funcs = {}

        for srctype, etype, dsttype in graph.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype])
            graph.nodes[srctype].data['Wh_{}'.format(etype)] = Wh
            funcs[etype] = (fn.copy_u('Wh_{}'.format(etype), 'm'), fn.mean('m', 'h'))

        graph.multi_update_all(funcs, 'sum')

        return {ntype: graph.nodes[ntype].data['h'] for ntype in graph.ntypes}