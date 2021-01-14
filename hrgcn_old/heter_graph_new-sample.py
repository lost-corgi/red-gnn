#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from torch.utils.data import DataLoader
import time

def plot_roc(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def get_f1_score(y_true, y_pred):
    """
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]
    :param y_true:
    :param y_pred:
    :return:
    """
    cf_m = confusion_matrix(y_true, y_pred)
    print("tn:",cf_m[0,0])
    print("fp:",cf_m[0,1])
    print("fn:",cf_m[1,0])
    print("tp:",cf_m[1,1])
    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1])
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall)


    return precision, recall, f1


def get_recall(y_true, y_pred):
    """
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]
    :param y_true:
    :param y_pred:
    :return:
    """
    cf_m = confusion_matrix(y_true, y_pred)

    # print(cf_m)
    return cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])


def get_auc_score(y_true, y_pred_prob):
    return roc_auc_score(y_true, y_pred_prob)


def plot_p_r_curve(y_true, y_pred_prob,best_logits,train_idx,val_idx):

    
    thresholds = [0]
    precision, recall, thresholds_2 = precision_recall_curve(y_true, y_pred_prob)
    
    #print('p',precision[0])
    #print('r',recall[0])
    #print('t',thresholds[0])
    
    #avp = average_precision_score(y_true, y_pred_prob)
    thresholds.extend(thresholds_2)

    
    gain_list = [] 
    
    thresholds_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    for each in thresholds_list:
        gain_list.append(torch.sum(torch.gt(best_logits,each))-torch.sum(torch.gt(best_logits[train_idx],each))
        -torch.sum(torch.gt(best_logits[val_idx],each)) )
    
    
    plt.plot(recall, precision, color='blue', lw=2, label='P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.show()
    plt.plot(thresholds, precision, color='blue', lw=2, label='Threshold-Precision Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Threshold-Precision Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.show()
    plt.plot(thresholds, recall, color='blue', lw=2, label='Threshold-Recall Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Threshold-Recall Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.show()
    plt.plot(thresholds_list, gain_list, color='blue', lw=1, label='Threshold-Gain Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Gain')
    plt.title('Threshold-Gain Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.show()

import networkx as nx
import pandas as pd
import s3fs
import pyarrow.parquet as pq
s3 = s3fs.S3FileSystem()
import numpy as np
import pyarrow as pa
import os
import networkx as nx 
import random
import dgl
import torch as th
import torch
import pandas as pd
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

device = "cpu"
def assign_a_gpu(gpu_no):
    device = torch.device("cuda:%s"%(str(gpu_no)) if torch.cuda.is_available() else "cpu")
    return device

device = assign_a_gpu(6)

print("device:",device)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })
        self.out_size = out_size

    def forward(self, graph, feat_dict):
        funcs = {}

      #  print("graph1:","graph")
      #  print("graph1:",graph.canonical_etypes)
        for srctype, etype, dsttype in graph.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype])
            graph.srcnodes[srctype].data['Wh_{}'.format(etype)] = Wh
            funcs[etype] = (fn.copy_u('Wh_{}'.format(etype), 'm'), fn.mean('m', 'h'))
       #     print("success")
         #   for i,k in funcs.items():
         #       print('func',i,k)

        graph.multi_update_all(funcs, 'sum')
        
        result = {}
        for ntype in graph.ntypes:
            if 'h' in graph.dstnodes[ntype].data:
                result[ntype] = graph.dstnodes[ntype].data['h']
            else:
                result[ntype] = torch.zeros(
                    graph.number_of_dst_nodes(ntype), self.out_size).to(device)

        return result

class HeteroRGCN(nn.Module):
    def __init__(self, blocks,target_node,node_feature,in_feats, h_dim, num_classes=2):
        super(HeteroRGCN, self).__init__()
        embed_dict = {ntype: torch.Tensor(blocks.number_of_nodes(ntype), in_feats).to(device) for ntype in blocks.ntypes}
        
            
#embed_dict = {ntype: nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device)) for ntype in graph.ntypes}
       
        for key, embed in embed_dict.items():
            embed_dict[key] = nn.init.zeros_(embed).to(device)
            #xavier_uniform_(embed) 如何可自动设置 

   #    embed_dict['user'] = nn.Parameter(graph.nodes['user'].data['f'].float())
        
            
        self.embed = embed_dict
        self.embed[target_node] = node_feature #这个如何保证

        self.layer1 = HeteroRGCNLayer(in_feats, h_dim, blocks.etypes)
        self.layer2 = HeteroRGCNLayer(h_dim, num_classes, blocks.etypes)


    def forward(self, blocks,graph,emb):
        if(blocks==None):
            h_dict = self.layer1(graph, emb)#self.embed)
            h_dict_2 = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = self.layer2(graph, h_dict_2)
            
        else:
            h_dict = self.layer1(blocks[0], emb)#self.embed)
            h_dict_2 = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = self.layer2(blocks[1], h_dict_2)
        return h_dict,h_dict_2   

def build_graph(relations_list,relations_data_list):
    relations_data_dic = {}
    i = 0 
    for each in relations_list:
        relations_data_dic[each] = relations_data_list[i]
        i += 1
    graph = dgl.heterograph(
       relations_data_dic
    )
    
    print('Node types:', graph.ntypes)
    print('Edge types:', graph.etypes)
    print('Canonical edge types:', graph.canonical_etypes)
    for each in graph.canonical_etypes:
        print('graph number edges--'+str(each)+':',graph.number_of_edges(each))
    for each in graph.ntypes:
        print('graph number nodes--'+str(each)+':',graph.number_of_nodes(each))
    return graph


def build_labels(pos_label_path,neg_label_path,label_column,pos_train_ratio,neg_train_ratio):
    
    
    pos_df = read_s3_file(pos_label_path) #,filesystem=s3).read().to_pandas()
    neg_df = read_s3_file(neg_label_path)# ,filesystem=s3).read().to_pandas() 
 

    pos_u_list = np.array(pos_df[pos_df['node_type'] == label_column]['no']) #获取某个点类型的pos标签
    neg_u_list = np.array(neg_df[neg_df['node_type'] == label_column]['no']) #获取某个点类型的neg标签
    print("neg_list:",np.max(neg_u_list))   
    print("pos_list:",np.max(pos_u_list))
    neg_split_pt = int(neg_u_list.shape[0] * neg_train_ratio) -1
    pos_split_pt = int(pos_u_list.shape[0] * pos_train_ratio) -1
    
    print("positive_samples_num:",pos_u_list.shape[0])
    print("negative_samples_num:",neg_u_list.shape[0])
    print("positive_train_samples_num:",pos_split_pt)
    print("negative_train_samples_num:",neg_split_pt)
    shuffled_neg = np.random.permutation(neg_u_list)
    train_neg_list = shuffled_neg[:neg_split_pt]
    valid_neg_list = shuffled_neg[neg_split_pt:]

    shuffled_pos = np.random.permutation(pos_u_list)
    train_pos_list = shuffled_pos[:pos_split_pt]
    valid_pos_list = shuffled_pos[pos_split_pt:]

    train_idx = np.concatenate([train_neg_list, train_pos_list])
    valid_idx = np.concatenate([valid_neg_list, valid_pos_list])

  
    neg_labels = np.zeros(neg_u_list.shape)
    pos_labels = np.ones(pos_u_list.shape)

    train_labels = np.concatenate([neg_labels[:neg_split_pt], pos_labels[:pos_split_pt]])
    valid_labels = np.concatenate([neg_labels[neg_split_pt:], pos_labels[pos_split_pt:]])

    train_idx = torch.from_numpy(train_idx).long()
    valid_idx = torch.from_numpy(valid_idx).long()
 
    train_labels = torch.Tensor(train_labels).long()
    valid_labels = torch.Tensor(valid_labels).long()
    
    return train_idx, valid_idx, train_labels, valid_labels



def train_model(net,g,train_mask,test_mask,epoch_num,lr_rate,label_column):
    import time
    import numpy as np
#g, features, labels, train_mask, test_mask = load_cora_data()
#features = th.FloatTensor(data.features)
#labels = th.LongTensor(data.labels)
#train_mask = th.BoolTensor(data.train_mask)
#test_mask = th.BoolTensor(data.test_mask)
#g = data.graph

    features = th.FloatTensor(g.ndata['feature'])
    labels = th.LongTensor(g.ndata['label'].long()).squeeze(1)
    #train_mask = th.BoolTensor(np.array(train_mask))
#    test_mask = th.BoolTensor(np.array(test_mask))
    optimizer = th.optim.Adam(net.parameters(), lr= lr_rate)
    dur = []
    penalty_lambda = 0.1
    for epoch in range(epoch_num):
        if epoch >=3:
            t0 = time.time()
    
        net.train()
        logits = net(g, features)
#when supervised case start
        g.ndata['hidden_feature'] = logits
        logp = F.log_softmax(logits, 1)
        #  g.apply_edges(func=diff_feature)
        loss = F.nll_loss(logp[train_mask], labels[train_mask]) 
        #+ penalty_lambda *th.mean(g.edata['edge_diff'].float())
#when supervised case finish
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        acc,tp,fp,tn,fn,p,r = evaluate(net, g, features, labels, test_mask,test_mask)
        print("Epoch {:05d} | Loss {:.4f} | tp {:.4f} | fp {:.4f} | tn {:.4f} | fn {:.4f} |  Precision {:.4f} | Recall {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(),tp,fp,tn,fn,p,r, np.mean(dur)))
    return net


relations_list = []
relations_data_list  = []
def read_s3_file(data_path): #, dir_name):
    s3 = s3fs.S3FileSystem()
    dir_path = '{}/'.format(data_path) #, dir_name)
    file_list = s3.ls(dir_path)
    dataset = ParquetFile(file_list,  open_with=s3.open)
    df = dataset.to_pandas()
    return df
from fastparquet import ParquetFile

path_1 =  's3://xhs.swap/user/hadoop/temp_s3/chuixue_user_use_relations_7_25_to_7_31'
relations_1 = read_s3_file(path_1)

relation_1_foward_edge = relations_1.node_1.values, relations_1.node_2.values
relation_1_back_edge = relations_1.node_2.values, relations_1.node_1.values
relations_list.append(('user', 'use', 'user_use'))
relations_list.append(('user_use', 'use by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


path_1 =  's3://xhs.swap/user/hadoop/temp_s3/chuixue_user_note_relations_7_25_to_7_31'
relations_1 = read_s3_file(path_1)
relation_1_foward_edge = relations_1.node_1.values, relations_1.node_2.values
relation_1_back_edge = relations_1.node_2.values, relations_1.node_1.values
relations_list.append(('user', 'interact', 'note'))
relations_list.append(('note', 'interact by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


graph_7_25_to_7_31 = build_graph(relations_list,relations_data_list)

import pickle
f = open('/apps/chuixue/graph_7_25_to_7_31', 'wb')
pickle.dump(graph_7_25_to_7_31, f)





#import pickle
#f = open('/apps/chuixue/graph_7_25_to_7_31', 'rb')
#graph_7_25_to_7_31 = pickle.load(f)


# In[13]:


pos_label_path =  's3://xhs.swap/user/hadoop/temp_s3/graph_black_label_7_25_to_7_31'
neg_label_path = 's3://xhs.swap/user/hadoop/temp_s3/graph_white_label_7_25_to_7_31_filter'
label_column = 'user'
pos_train_ratio = 0.8 #0.8  #0.8
neg_train_ratio = 0.9 #0.9  #0.9 #0.8
train_idx_7_25_to_7_31, val_idx_7_25_to_7_31, train_labels_7_25_to_7_31, valid_labels_7_25_to_7_31 = build_labels(pos_label_path,neg_label_path,label_column,pos_train_ratio,neg_train_ratio)

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import RobustScaler

def node_feature_handle(df,categorical_variables_list,numerical_variables_list):
    df = df[categorical_variables_list + numerical_variables_list ]
    for each in categorical_variables_list:
        print(each)
        features_encoder = LabelBinarizer()
        features_encoder.fit(df[each])
        transformed = features_encoder.transform(df[each])
        ohe_df = pd.DataFrame(transformed)    
        df = pd.concat([df, ohe_df], axis=1)  
#   categorical_variables_list.append('no')
    df = df.drop(categorical_variables_list, axis=1)
    df.info()
    scaler = RobustScaler()
    df[numerical_variables_list] = scaler.fit_transform(df[numerical_variables_list])
    torch_tensor = torch.tensor(df.values)
    return torch_tensor


path_features =  's3://xhs.swap/user/hadoop/temp_s3/node_feature_7_25_to_7_31'
user_nodes_features_pd_7_25_to_7_31 = read_s3_file(path_features)#,filesystem=s3).read().to_pandas()

# In[16]:


user_nodes_features_pd_7_25_to_7_31.columns


# In[17]:


#cate_list = [ 'platform', 'register_channel', 'tenure', 'member_flag']
cate_list = []
num_list  = [ 'days_active_in_l56d', 'days_active_in_l28d', 'days_active_in_l14d', 'days_active_in_l7d',
             'days_active_in_l3d','frequent_city_days_90d']
user_nodes_features_pd_7_25_to_7_31[cate_list] = user_nodes_features_pd_7_25_to_7_31[cate_list].fillna('null')
user_nodes_features_pd_7_25_to_7_31[num_list] = user_nodes_features_pd_7_25_to_7_31[num_list].fillna(-100)


# In[18]:


user_nodes_features_pd_7_25_to_7_31.info()


user_node_feature_7_25_to_7_31 = node_feature_handle(user_nodes_features_pd_7_25_to_7_31,cate_list,num_list)

in_feats = user_node_feature_7_25_to_7_31.shape[1] #need change according to input feature 
h_dim = 16
num_classes = 2
net = HeteroRGCN(graph_7_25_to_7_31,'user',user_node_feature_7_25_to_7_31.float(), in_feats, h_dim, num_classes).to(device=device) #有两个版本的feature为0 和feature随机初始化的
max_epoch = 300

print(user_node_feature_7_25_to_7_31.shape[1])

def train_base_graph_model(net,graph,MAX_EPOCH,label_column,train_idx, val_idx, train_labels, valid_labels):

    train_labels = train_labels.to(device=device)
    valid_labels = valid_labels.to(device=device)
    best_val_acc = 0
    best_logits =[]
    best_preds = []
    opt = torch.optim.Adam(net.parameters(), lr=0.001)  #, weight_decay=5e-4)
    for epoch in range(MAX_EPOCH):
        logits,embed_vectors = net(graph)#[label_column]
        logits = logits[label_column]
        logits = F.softmax(logits,dim = 1)
        loss = F.mse_loss(logits[train_idx][:, 1], train_labels.float())
        
        net.train()

   #     loss = F.cross_entropy(logits[train_idx], train_labels)
        
        pred = logits.argmax(1)  # dim = 1 along the row 
        train_acc = (pred[train_idx] == train_labels).float().mean()
        val_acc = (pred[val_idx] == valid_labels).float().mean()

        opt.zero_grad() 
        loss.backward()
        opt.step()
        
        if epoch % 5 == 0:
            # print("epoch {:0>3}/{:0>3} \t loss: {:.4f}".format(epoch, MAX_EPOCH, loss.item()))
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f ' % (
                loss.item(),
                train_acc.item(),
                val_acc.item()
                #.item(),
            ))

    best_val_acc = val_acc
    best_logits = logits.cpu().detach().numpy()
    best_preds = pred.cpu().detach().numpy()     
    y_true = valid_labels.cpu().detach().numpy()
    y_pred = best_preds[val_idx]
    y_pred_prob = best_logits[val_idx]

    extra_sum = torch.sum(pred) - torch.sum(pred[train_idx])-torch.sum(pred[val_idx])
    print(torch.sum(pred))
    print(torch.sum(pred[train_idx]))
    print(torch.sum(pred[val_idx]))
    print('extra_sum:',extra_sum.item())
    best_precision, best_recall, best_f1 = get_f1_score(y_true, y_pred)
    best_auc = get_auc_score(y_true, y_pred_prob[:, 1])
    plot_p_r_curve(y_true, y_pred_prob[:, 1],logits[:,1],train_idx,val_idx)
    
    
    return net,best_precision, best_recall, best_f1, best_auc


def extract_embed(node_embed, block):
    emb = {}
    for ntype in block.srctypes:
        nid = block.srcnodes[ntype].data[dgl.NID]
        emb[ntype] = node_embed[ntype][nid]
    return emb


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

def load_subtensor(g, labels, seeds,category, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes[category]].to(device)
    batch_labels = labels[seeds[category]].to(device)
    return batch_inputs, batch_labels

#input_nodes = {ntype: blocks[0].srcnodes[ntype].data[dgl.NID] for ntype in blocks[0].srctypes}
#seeds =  {ntype: blocks[-1].dstnodes[ntype].data[dgl.NID] for ntype in blocks[-1].dsttypes}
           

def load_subtensor_heter(g, labels, seeds,category, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    
 
    #  lbl = labels[seeds[category]]
    #  if use_cuda:
    #  emb = {k : e.cuda() for k, e in emb.items()}
    #  lbl = lbl.cuda()
    
  #  print('seeds_info',seeds[category])
    
    
    batch_inputs = {k: e.to(device) for k,e in input_nodes.items() }
    
    
 #   batch_inputs = g.ndata['features'][input_nodes[category]].to(device)
    batch_labels = labels.to(device)
    return batch_inputs, batch_labels    


    

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])


# In[ ]:





# In[27]:


class HeteroNeighborSampler:
    """Neighbor sampler on heterogeneous graphs
    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph
    category : str
        Category name of the seed nodes.
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, category, fanouts):
        self.g = g
        self.category = category
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        log = open('log.txt', 'w')
        import datetime
        print('Datetime:', datetime.datetime.now(), file=log)

        blocks = []
        seeds = {self.category : th.tensor(seeds).long()}
        cur = seeds

        print('Seed input', file=log)
        for ntype, nid in cur.items():
            print(ntype + ':', file=log)
            np.savetxt(log, [nid.numpy()], fmt='%ld')

        for fanout in self.fanouts:
            if fanout is None:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)

            print('Frontier edges', file=log)
            frontier_edges = {etype: frontier.all_edges(order='eid', etype=etype) for etype in frontier.canonical_etypes}
            for etype, (u, v) in frontier_edges.items():
                print(str(etype) + ':', file=log)
                np.savetxt(log, [u.numpy()], fmt='%ld')
                np.savetxt(log, [v.numpy()], fmt='%ld')

            block = dgl.to_block(frontier, cur)
            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)

            print('Block edges', file=log)
            block_edges = {etype: block.all_edges(order='eid', etype=etype) for etype in block.canonical_etypes}
            for etype, (u, v) in block_edges.items():
                print(str(etype) + ':', file=log)
                np.savetxt(log, [u.numpy()], fmt='%ld')
                np.savetxt(log, [v.numpy()], fmt='%ld')
                np.savetxt(log, [block.edges[etype].data[dgl.EID].numpy()], fmt='%ld')
            print('Block src nodes', file=log)
            for ntype in block.srctypes:
                print(ntype + ':', file=log)
                np.savetxt(log, [block.srcnodes[ntype].data[dgl.NID].numpy()], fmt='%ld')
            print('Block dst nodes', file=log)
            for ntype in block.dsttypes:
                print(ntype + ':', file=log)
                np.savetxt(log, [block.dstnodes[ntype].data[dgl.NID].numpy()], fmt='%ld')

        log.close()

        return seeds, blocks



import tqdm
import gc



def evaluate(model,predict_node_type, graph, labels, label_mask):
    model.eval()
    net.embed = {k: e.to(device) for k,e in net.embed.items() }
    node_embed  = net.embed
    with th.no_grad():
        logits,embed_vectors = model(None,graph,node_embed)#[label_column]
        logits = logits[predict_node_type]
        logits = logits[label_mask]
     #  labels = labels[label_mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        TP = 0
        TP += ((indices == 1) & (labels == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN = 0 
        TN += ((indices == 0) & (labels == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN = 0
        FN += ((indices == 0) & (labels == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP = 0
        FP += ((indices == 1) & (labels == 0)).cpu().sum()
        p = TP.item()/(FP.item()+TP.item()+1.0)
        r = TP.item() / (TP.item() + FN.item()+1.0)
        return correct.item() * 1.0 / len(labels),TP,FP,TN,FN,p,r


# In[30]:


def train_base_graph_model_blocks(net,graph,MAX_EPOCH,label_column,train_idx, val_idx, train_labels, valid_labels):

    train_labels = train_labels.to(device=device)
    valid_labels = valid_labels.to(device=device)
    best_val_acc = 0
    best_logits =[]
    best_preds = []
    opt = torch.optim.Adam(net.parameters(), lr=0.001)  #, weight_decay=5e-4)
    
   # sampler = NeighborSampler(graph, [100,200]) # sample fan outs

    category = label_column
    sampler = HeteroNeighborSampler(graph, category, [10,20])
     
    net = net.to(device)
    net.embed = {k: e.to(device) for k,e in net.embed.items() }
    node_embed  = net.embed
    
    dataloader = DataLoader(
        dataset= train_idx,  
        
        batch_size= 64, #seeds number 128 
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers= 4 ) #16
    
    train_idx_dic = {x.item():i for i,x in enumerate(train_idx)}

    k = 0
    for i,j in train_idx_dic.items():
        print(i,j)
        k += 1
        if(k > 10):
            break
    
    avg = 0
    iter_tput = []
    for epoch in range(MAX_EPOCH):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (seeds,blocks) in tqdm.tqdm_notebook(
            enumerate(dataloader), total=len(dataloader)):
            tic_step = time.time()
            blocks = [each.to(device) for each in blocks]
            
            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            
            input_nodes = {ntype: blocks[0].srcnodes[ntype].data[dgl.NID] for ntype in blocks[0].srctypes}
            seeds =  {ntype: blocks[-1].dstnodes[ntype].data[dgl.NID] for ntype in blocks[-1].dsttypes}
           # print
           # print("seeds_tensor:",seeds[category])
        
            emb = extract_embed(node_embed, blocks[0])
           # input_nodes = blocks[0].srcdata[dgl.NID][category]
           # seeds = blocks[-1].dstdata[dgl.NID]
             
            emb = {k : e.to(device) for k, e in emb.items()}
            # Load the input features as well as output labels
            seeds_labels = train_labels[torch.tensor([ train_idx_dic[x.item()] for x in seeds[category]])]
           # print("seeds_labels",seeds_labels)
            batch_inputs, batch_labels = load_subtensor_heter(graph,seeds_labels,seeds,category, input_nodes, device)  

            # Compute loss and prediction 
            #batch_pred = net(blocks, batch_inputs) batch_inputs 是点的feature 这里要变化model
            logits,embed_vectors = net(blocks,None,emb) #,batch_inputs)#[label_column]
            logits = logits[label_column] #这里要变化  
            logits = F.softmax(logits,dim = 1)
            loss = F.mse_loss(logits[:, 1], batch_labels.float())
            net.train()
#     loss = F.cross_entropy(logits[train_idx], train_labels) 不采用信息熵
            pred = logits.argmax(1)  # dim = 1 along the row 
            train_acc = (pred == batch_labels).float().mean()
            #val_acc = (pred[val_idx] == valid_labels).float().mean()
            opt.zero_grad() 
            loss.backward()
            opt.step() 
            
            
            #loss = loss_fcn(batch_pred, batch_labels)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            #if step % 5  == 0:
              #  acc = compute_acc(batch_pred, batch_labels)
             #   gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
             #   print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
             #       epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
              
        if epoch % 5 == 0:
            # print("epoch {:0>3}/{:0>3} \t loss: {:.4f}".format(epoch, MAX_EPOCH, loss.item()))
            print('Loss %.4f, Train Acc %.4f ' % (
                loss.item(),
                train_acc.item()
             #   val_acc.item()
                #.item(),
            ))
            
    
        gc.collect()
    #迭代完了加一个evlaution
    #emb = extract_embed(node_embed, graph)
    #emb = {k : e.to(device) for k, e in emb.items()}
    
    
   # correct_ratio,TP,FP,TN,FN,p,r = evaluate(net,label_column, graph, node_embed,valid_labels, val_idx)
   # print('acc_ratio %.2f, TP %.2f,FP %.2f, TN %.2f,FN %.2f, p %.2f,r %.2f  ' % (
    # correct_ratio,TP,FP,TN,FN,p,r
   #))


#    best_val_acc = val_acc
#    best_logits = logits.cpu().detach().numpy()
#    best_preds = pred.cpu().detach().numpy()     
#    y_true = valid_labels.cpu().detach().numpy()
#    y_pred = best_preds[val_idx]
#    y_pred_prob = best_logits[val_idx]

#    extra_sum = torch.sum(pred) - torch.sum(pred[train_idx])-torch.sum(pred[val_idx])
#    print(torch.sum(pred))
#    print(torch.sum(pred[train_idx]))
#    print(torch.sum(pred[val_idx]))
#    print('extra_sum:',extra_sum.item())
#    best_precision, best_recall, best_f1 = get_f1_score(y_true, y_pred)
#    best_auc = get_auc_score(y_true, y_pred_prob[:, 1])
#    plot_p_r_curve(y_true, y_pred_prob[:, 1],logits[:,1],train_idx,val_idx)
    
    
    return net #,best_precision, best_recall, best_f1, best_auc


# In[35]:


#device = assign_a_gpu(6)
max_epoch = 15
new_net = train_base_graph_model_blocks(net,graph_7_25_to_7_31,max_epoch,label_column,train_idx_7_25_to_7_31, val_idx_7_25_to_7_31, train_labels_7_25_to_7_31, valid_labels_7_25_to_7_31)
th.save(new_net.state_dict(), "/apps/chuixue/gcn_model_sample_new_7_25_to_7_31.pt")


device = "cpu"
net_new = HeteroRGCN(graph_7_25_to_7_31,'user',user_node_feature_7_25_to_7_31.float(), in_feats, h_dim, num_classes).to(device=device) #有两个版本的feature为0 和feature随机初始化的
net_new.load_state_dict(th.load("/apps/chuixue/gcn_model_sample_new_7_25_to_7_31.pt"))
correct_ratio,TP,FP,TN,FN,p,r = evaluate(net_new,label_column, graph_7_25_to_7_31,valid_labels_7_25_to_7_31, val_idx_7_25_to_7_31)
print('acc_ratio %.2f, TP %.2f,FP %.2f, TN %.2f,FN %.2f, p %.2f,r %.2f  ' % (correct_ratio,TP,FP,TN,FN,p,r))

