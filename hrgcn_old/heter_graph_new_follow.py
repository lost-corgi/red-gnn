#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


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
    plt.savefig("gcn_pr.png")
    plt.close()
    plt.plot(thresholds, precision, color='blue', lw=2, label='Threshold-Precision Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Threshold-Precision Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.savefig("gcn_tp.png")
    plt.close()
    plt.plot(thresholds, recall, color='blue', lw=2, label='Threshold-Recall Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Threshold-Recall Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.savefig("gcn_tr.png")
    plt.close()
    plt.plot(thresholds_list, gain_list, color='blue', lw=1, label='Threshold-Gain Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Gain')
    plt.title('Threshold-Gain Curve for Binary Classification')
    plt.legend(loc="top right")
    plt.savefig("gcn_tg.png")




# In[4]:


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
import gc

device = "cpu"
def assign_a_gpu(gpu_no):
    device = torch.device("cuda:%s"%(str(gpu_no)) if torch.cuda.is_available() else "cpu")
    return device


# In[ ]:





# In[5]:


#device = assign_a_gpu(1)


# In[6]:



# In[7]:


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


# In[8]:


class HeteroRGCN(nn.Module):
    def __init__(self, graph,target_node,node_feature,in_feats, h_dim, num_classes=2):
        super(HeteroRGCN, self).__init__()

#       embed_dict = {ntype: nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device)) for ntype in graph.ntypes}
        embed_dict = {ntype: torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device) for ntype in graph.ntypes}

        for key, embed in embed_dict.items():
            embed_dict[key] = nn.init.zeros_(embed)
            #xavier_uniform_(embed) 如何可自动设置 

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
#add softmax here 
        return h_dict,h_dict_2    #['user'] #改成通用的 


# In[ ]:





# In[9]:


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

#('user', 'collect', 'note')


# In[10]:


def build_labels(pos_label_path,neg_label_path,label_column,pos_train_ratio,neg_train_ratio):
    
    pos_df = pq.ParquetDataset(pos_label_path,filesystem=s3).read().to_pandas()
    neg_df = pq.ParquetDataset(neg_label_path,filesystem=s3).read().to_pandas() 

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


# In[11]:


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


# In[12]:


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


path_1 =  's3://xhs.swap/user/hadoop/temp_s3/chuixue_user_use_relations_8_18_to_8_20'
relations_1 = read_s3_file(path_1)
relation_1_foward_edge = relations_1.node_1.values, relations_1.node_2.values
relation_1_back_edge = relations_1.node_2.values, relations_1.node_1.values
relations_list.append(('user', 'use', 'user_use'))
relations_list.append(('user_use', 'use by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


path_1 =  's3://xhs.swap/user/hadoop/temp_s3/chuixue_user_note_relations_8_18_to_8_20'
relations_1 = read_s3_file(path_1)
relation_1_foward_edge = relations_1.node_1.values, relations_1.node_2.values
relation_1_back_edge =  relations_1.node_2.values, relations_1.node_1.values
relations_list.append(('user', 'interact', 'note'))
relations_list.append(('note', 'interacted by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


path_1 =  's3://xhs.swap/user/hadoop/temp_s3/chuixue_user_user_relations_8_18_to_8_20'
relations_1 = read_s3_file(path_1)
relation_1_foward_edge = relations_1.node_1.values, relations_1.node_2.values
relation_1_back_edge =  relations_1.node_2.values, relations_1.node_1.values
relations_list.append(('user', 'follow', 'user'))
relations_list.append(('note', 'followed by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


graph_8_18_to_8_20 = build_graph(relations_list,relations_data_list)


# In[13]:


import pickle

f = open('/apps/chuixue/graph_8_18_to_8_20', 'wb')
pickle.dump(graph_8_18_to_8_20, f)


# In[8]:


#import pickle
#f = open('/apps/chuixue/graph_8_18_to_8_20', 'rb')
#graph_8_18_to_8_20 = pickle.load(f)


# In[14]:


pos_label_path =  's3://xhs.swap/user/hadoop/temp_s3/graph_black_label_8_18_to_8_20'
neg_label_path = 's3://xhs.swap/user/hadoop/temp_s3/graph_white_label_8_18_to_8_20_filter'
label_column = 'user'
category = 'user'
pos_train_ratio = 0.8 #0.8  #0.8
neg_train_ratio = 0.9 #0.9  #0.9 #0.8
train_idx_8_18_to_8_20, val_idx_8_18_to_8_20, train_labels_8_18_to_8_20, valid_labels_8_18_to_8_20 = build_labels(pos_label_path,neg_label_path,label_column,pos_train_ratio,neg_train_ratio)


# In[15]:


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

#feature_list =  [3,3,3]


# In[16]:


path_features =  's3://xhs.swap/user/hadoop/temp_s3/node_feature_8_18_to_8_20'
user_nodes_features_pd_8_18_to_8_20 = pq.ParquetDataset(path_features,filesystem=s3).read().to_pandas()


# In[17]:


user_nodes_features_pd_8_18_to_8_20.columns


# In[18]:


user_nodes_features_pd_8_18_to_8_20.shape


# In[19]:


#cate_list = [ 'platform', 'register_channel', 'tenure', 'member_flag']
cate_list = []
num_list  = [ 'days_active_in_l56d', 'days_active_in_l28d', 'days_active_in_l14d', 'days_active_in_l7d',
             'days_active_in_l3d','frequent_city_days_90d']
user_nodes_features_pd_8_18_to_8_20[cate_list] = user_nodes_features_pd_8_18_to_8_20[cate_list].fillna('null')
user_nodes_features_pd_8_18_to_8_20[num_list] = user_nodes_features_pd_8_18_to_8_20[num_list].fillna(-100)


# In[20]:


user_nodes_features_pd_8_18_to_8_20.info()


# In[21]:


user_node_feature_8_18_to_8_20 = node_feature_handle(user_nodes_features_pd_8_18_to_8_20,cate_list,num_list)


# In[22]:


in_feats = user_node_feature_8_18_to_8_20.shape[1] #need change according to input feature 
h_dim = 16
num_classes = 2
net = HeteroRGCN(graph_8_18_to_8_20,'user',user_node_feature_8_18_to_8_20.float(), in_feats, h_dim, num_classes).to(device=device) #有两个版本的feature为0 和feature随机初始化的
max_epoch = 300


# In[17]:


#net.load_state_dict(th.load("/apps/chuixue/gcn_model_heter_online_predict_3_20_to_3_23_temp1.pt"))


# In[18]:


#赋予feature 
#也可以考虑一个list 有feature点 ，以及对应的feature list

#net.embed['user'] = nn.Parameter(user_node_feature.float()) 


# In[23]:


#user_node_feature_8_18_to_8_20.shape[1]


# In[24]:


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

        gc.collect()

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


# In[ ]:


max_epoch = 10
net,best_precision, best_recall, best_f1, best_auc = train_base_graph_model(net,graph_8_18_to_8_20,
max_epoch,label_column,train_idx_8_18_to_8_20, val_idx_8_18_to_8_20, train_labels_8_18_to_8_20, valid_labels_8_18_to_8_20)


# In[25]:


print(best_precision, best_recall, best_f1, best_auc)


# In[ ]:


th.save(net.state_dict(), "/apps/chuixue/gcn_model_heter_8_18_to_8_20.pt")


# In[26]:
