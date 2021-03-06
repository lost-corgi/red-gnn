import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl
import tqdm
import layers
import dgl.nn as dglnn

class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, activation, dropout, rel_names):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.layers = nn.ModuleList()
        #i2h
        self.layers.append(dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_dim, h_dim) for rel in rel_names}))
        #h2h
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv(
                {rel: dglnn.GraphConv(h_dim, h_dim) for rel in rel_names}))
        #h2o
        self.layers.append(dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(h_dim, out_dim) for rel in rel_names}))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = {k: v[:block.num_dst_nodes(k)] for k, v in h.items()}
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = {k: self.activation(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        for l, layer in enumerate(self.layers):
            y = {
                k: torch.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h_dst = {k: v[:block.num_dst_nodes(k)] for k, v in h.items()}
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = {k: self.activation(v) for k, v in h.items()}
                    h = {k: self.dropout(v) for k, v in h.items()}

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y

# class SAGE(nn.Module):
#     """
#     GraphSAGE with mean AGG and dropout, training should use mini-batch, i.e. dataloader with blocks
#     """
#     def __init__(self,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super().__init__()
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.layers = nn.ModuleList()
#         self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
#         for i in range(1, n_layers - 1):
#             self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
#         self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation
#
#     def forward(self, blocks, x):
#         h = x
#         for l, (layer, block) in enumerate(zip(self.layers, blocks)):
#             # We need to first copy the representation of nodes on the RHS from the
#             # appropriate nodes on the LHS.
#             # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
#             # would be (num_nodes_RHS, D)
#             h_dst = h[:block.num_dst_nodes()]
#             # Then we compute the updated representation on the RHS.
#             # The shape of h now becomes (num_nodes_RHS, D)
#             h = layer(block, (h, h_dst))
#             if l != len(self.layers) - 1:
#                 h = self.activation(h)
#                 h = self.dropout(h)
#         return h
#
#     def inference(self, g, x, device):
#         """
#         Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
#         g : the entire graph.
#         x : the input of entire node set.
#         The inference code is written in a fashion that it could handle any number of nodes and
#         layers.
#         """
#         # During inference with sampling, multi-layer blocks are very inefficient because
#         # lots of computations in the first few layers are repeated.
#         # Therefore, we compute the representation of all nodes layer by layer.  The nodes
#         # on each layer are of course splitted in batches.
#         # TODO: can we standardize this?
#         for l, layer in enumerate(self.layers):
#             y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)
#
#             sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
#             dataloader = dgl.dataloading.NodeDataLoader(
#                 g,
#                 torch.arange(g.num_nodes()),
#                 sampler,
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 drop_last=False,
#                 num_workers=args.num_workers)
#
#             for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
#                 block = blocks[0].int().to(device)
#
#                 h = x[input_nodes]
#                 h_dst = h[:block.num_dst_nodes()]
#                 h = layer(block, (h, h_dst))
#                 if l != len(self.layers) - 1:
#                     h = self.activation(h)
#                     h = self.dropout(h)
#
#                 y[output_nodes] = h
#
#             x = y
#         return y

class HeteroRGCN(nn.Module):
    """
    inefficient full batch(graph) forward, applicable only to small graphs
    """
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = layers.HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = layers.HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict

# # chuixue's version with fixed input feature, target node init to input_feature, other type of node to 0
# class HeteroRGCN(nn.Module):
#     def __init__(self, graph, target_node, node_feature, in_feats, h_dim, num_classes=2):
#         super(HeteroRGCN, self).__init__()
#
#
#         embed_dict = {ntype: torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device) for ntype in graph.ntypes}
#
#         for key, embed in embed_dict.items():
#             embed_dict[key] = nn.init.zeros_(embed)
#
#         #    embed_dict['user'] = nn.Parameter(graph.nodes['user'].data['f'].float())
#
#         self.embed = embed_dict
#         self.embed[target_node] = node_feature
#
#         self.layer1 = layers.HeteroRGCNLayer(in_feats, h_dim, graph.etypes)
#         self.layer2 = layers.HeteroRGCNLayer(h_dim, num_classes, graph.etypes)
#
#     def forward(self, graph):
#         h_dict = self.layer1(graph, self.embed)
#         h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
#         h_dict = self.layer2(graph, h_dict)
#         return h_dict
