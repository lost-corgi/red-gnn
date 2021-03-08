import argparse
import numpy as np
import torch
import dgl
import s3fs
import pyarrow.parquet as pq
import torch.multiprocessing as mp
from hgraph_builder import *
from train import *
from data_utils import sample_label


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Train user-device graph")
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--sample-ratio', type=int, default=3)
    argparser.add_argument('--num-epochs', type=int, default=100)
    # argparser.add_argument('--input-dim', type=int, default=10)
    argparser.add_argument('--hidden-dim', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=5)
    argparser.add_argument('--fan-out', type=str, default='10,10,10,10,10')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--is-pad', type=bool, default=True)
    argparser.add_argument('--/', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    # argparser.add_argument('--loss-func', type=str, default='BCELoss')
    # argparser.add_argument('--use-label-subgraph', type=bool, default=True)

    argparser.add_argument('--user-table', type=str,
                           default='dm_as_gnn_user_interact_note_user_normalized_feature_1d_inc')
    argparser.add_argument('--device-table', type=str, default='dm_as_gnn_user_interact_note_device_feature_1d_inc')
    argparser.add_argument('--relation-table', type=str, default='dm_as_gnn_user_interact_note_device_relation_1d_inc')
    argparser.add_argument('--label-table', type=str, default='dm_as_gnn_user_interact_note_user_label_1d_inc')
    argparser.add_argument('--label-entity', type=str, default='user')
    argparser.add_argument('--dsnodash', type=str, default='20210306')
    # parser.add_argument('output_path', type=str)
    args = argparser.parse_args()
    s3 = s3fs.S3FileSystem()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    user_table_path = 's3://xhs.alpha/reddm/' + args.user_table + '/dtm=%s' % args.dsnodash
    user_features = pq.ParquetDataset(user_table_path, filesystem=s3).read().to_pandas()

    device_table_path = 's3://xhs.alpha/reddm/' + args.device_table + '/dtm=%s' % args.dsnodash
    device_features = pq.ParquetDataset(device_table_path, filesystem=s3).read().to_pandas()

    relation_table_path = 's3://xhs.alpha/reddm/' + args.relation_table + '/dtm=%s' % args.dsnodash
    relation_df = pq.ParquetDataset(relation_table_path, filesystem=s3).read().to_pandas()

    label_table_path = 's3://xhs.alpha/reddm/' + args.label_table + '/dtm=%s' % args.dsnodash
    labels = pq.ParquetDataset(label_table_path, filesystem=s3).read().to_pandas()

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_features, 'user_entity_id', 'user')
    graph_builder.add_entities(device_features, 'device_entity_id', 'device')
    graph_builder.add_binary_relations(relation_df, 'user_entity_id', 'device_entity_id', 'used')
    graph_builder.add_binary_relations(relation_df, 'device_entity_id', 'user_entity_id', 'used-by')

    g = graph_builder.build()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()

    # construct subgraph for labeled computation probably save some memory of storing features
    # g = construct_computation_graph(g, n_layers, labels[label_entity_col_name].values, label_entity_type)

    # Assign features.
    user_features = user_features.sort_values(by='user_entity_id').values[:, 1:]
    device_features = device_features.sort_values(by='device_entity_id').values[:, 1:]
    labels = labels.values
    pos_label_count = np.count_nonzero(labels[:, 1] > 0)
    neg_labels = labels[labels[:, 1] == 0]
    neg_labels = neg_labels[np.random.randint(neg_labels.shape[0], size=pos_label_count*args.sample_ratio), :]
    labels = np.vstack((labels[labels[:, 1] > 0], neg_labels))

    val_num, test_num = labels.shape[0] // 10, labels.shape[0] // 10
    n_classes = labels[:, 1].max() + 1
    num_user_feature = user_features.shape[1]
    num_device_feature = device_features.shape[1]

    np.random.shuffle(labels)
    train_idx, val_idx, test_idx = torch.from_numpy(labels[val_num + test_num:, 0]),\
                                   torch.from_numpy(labels[:val_num, 0]), torch.from_numpy(labels[val_num:val_num + test_num, 0])
    expand_labels = np.empty(user_features.shape[0], dtype=np.float32)
    expand_labels[labels[:, 0]] = labels[:, 1]
    labels = torch.from_numpy(expand_labels)
    labels = torch.unsqueeze(labels, 1)
    np.savez_compressed('./dataset/1d%s' % args.dsnodash, user_f=user_features, device_f=device_features, labels=labels)
    dgl.save_graphs('./dataset/1d_%s_graph' % args.dsnodash, [g])
    #
    # user_features = F.pad(torch.tensor(user_features, device=device, dtype=torch.float32), (0, num_device_feature))
    # device_features = F.pad(torch.tensor(device_features, device=device, dtype=torch.float32), (num_user_feature, 0))
    user_features = torch.from_numpy(user_features).type(torch.float32)     # user_features too large to fit in gpu memory
    device_features = torch.from_numpy(device_features).type(torch.float32)
    entity_features = {'user': user_features, 'device': device_features}

    # g.edges['used'].data['weights'] = torch.ShortTensor(relation_df['relation_edge_weight'].values)
    # g.edges['used-by'].data['weights'] = torch.ShortTensor(relation_df['relation_edge_weight'].values)
    del relation_df

    # prepare for training
    data = train_idx, val_idx, test_idx, num_user_feature + num_device_feature, num_user_feature, num_device_feature, \
           labels, n_classes, entity_features, g

    if n_gpus == 1:
        train_mp(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=thread_wrapped_func(train_mp),
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
