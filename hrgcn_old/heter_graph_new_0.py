#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


import pandas as pd
import s3fs
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
import os
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

s3 = s3fs.S3FileSystem()

device = "cpu"
def assign_a_gpu(gpu_no):
    device = torch.device("cuda:%s"%(str(gpu_no)) if torch.cuda.is_available() else "cpu")
    return device

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


pos_label_path =  's3://xhs.swap/user/hadoop/temp_s3/graph_black_label_4_28_to_4_30'
neg_label_path = 's3://xhs.swap/user/hadoop/temp_s3/graph_white_label_4_28_to_4_30_filter'
label_column = 'user'
pos_train_ratio = 0.8 #0.8  #0.8
neg_train_ratio = 0.9 #0.9  #0.9 #0.8
train_idx_4_28_to_4_30, val_idx_4_28_to_4_30, train_labels_4_28_to_4_30, valid_labels_4_28_to_4_30 = build_labels(pos_label_path,neg_label_path,label_column,pos_train_ratio,neg_train_ratio)



path_features =  's3://xhs.swap/user/hadoop/temp_s3/node_feature_4_28_to_4_30'
user_nodes_features_pd_4_28_to_4_30 = pq.ParquetDataset(path_features,filesystem=s3).read().to_pandas()


#cate_list = [ 'platform', 'register_channel', 'tenure', 'member_flag']
cate_list = []
num_list  = [ 'days_active_in_l56d', 'days_active_in_l28d', 'days_active_in_l14d', 'days_active_in_l7d',
             'days_active_in_l3d','frequent_city_days_90d']
user_nodes_features_pd_4_28_to_4_30[cate_list] = user_nodes_features_pd_4_28_to_4_30[cate_list].fillna('null')
user_nodes_features_pd_4_28_to_4_30[num_list] = user_nodes_features_pd_4_28_to_4_30[num_list].fillna(-100)


print(user_nodes_features_pd_4_28_to_4_30.info())

user_node_feature_4_28_to_4_30 = node_feature_handle(user_nodes_features_pd_4_28_to_4_30,cate_list,num_list)



in_feats = user_node_feature_4_28_to_4_30.shape[1] #need change according to input feature 
h_dim = 16
num_classes = 2
net = HeteroRGCN(graph_4_28_to_4_30,'user',user_node_feature_4_28_to_4_30.float(), in_feats, h_dim, num_classes).to(device=device) #有两个版本的feature为0 和feature随机初始化的
max_epoch = 300


# In[17]:


#net.load_state_dict(th.load("/apps/chuixue/gcn_model_heter_online_predict_4_28_to_4_30_temp1.pt"))


# In[18]:


#赋予feature 
#也可以考虑一个list 有feature点 ，以及对应的feature list

#net.embed['user'] = nn.Parameter(user_node_feature.float()) 


# In[19]:


#user_node_feature_4_28_to_4_30.shape[1]


# In[20]:


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


max_epoch = 60
net,best_precision, best_recall, best_f1, best_auc = train_base_graph_model(net,graph_4_28_to_4_30,max_epoch,label_column,train_idx_4_28_to_4_30, val_idx_4_28_to_4_30, train_labels_4_28_to_4_30, valid_labels_4_28_to_4_30)
th.save(net.state_dict(), "/apps/chuixue/gcn_model_heter_online_predict_4_28_to_4_30_temp1.pt")
                                         
print(best_precision, best_recall, best_f1, best_auc)
