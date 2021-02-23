import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from hgraph_builder import *
import s3fs
import pyarrow.parquet as pq
from dateutil.parser import parse as dt_parse
from train import train
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

s3 = s3fs.S3FileSystem()

# dsnodash = dt_parse(args.date_key).strftime('%Y%m%d')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Train user-device graph")
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=100)
    # argparser.add_argument('--input-dim', type=int, default=10)
    argparser.add_argument('--hidden-dim', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--is-pad', type=bool, default=True)
    # argparser.add_argument('--loss-func', type=str, default='BCELoss')  # TODO: modify train.py to take args.loss_func
    # argparser.add_argument('--use-label-subgraph', type=bool, default=True)

    argparser.add_argument('--user-table', type=str,
                           default='dm_as_gnn_user_interact_note_user_normalized_feature_1d_inc')
    argparser.add_argument('--device-table', type=str, default='dm_as_gnn_user_interact_note_device_feature_1d_inc')
    argparser.add_argument('--relation-table', type=str, default='dm_as_gnn_user_interact_note_device_relation_1d_inc')
    argparser.add_argument('--label-table', type=str, default='dm_as_gnn_user_interact_note_user_label_1d_inc')
    argparser.add_argument('--label-entity', type=str, default='user')
    argparser.add_argument('--dsnodash', type=str, default='20210206')
    # parser.add_argument('output_path', type=str)
    args = argparser.parse_args()

    # output_path = args.output_path

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

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
    val_num, test_num = labels.shape[0] // 8, labels.shape[0] // 8
    n_classes = labels[:, 1].max() + 1
    num_user_feature = user_features.shape[1]
    num_device_feature = device_features.shape[1]

    # if args.use_label_subgraph:
    #     idxs = np.arange(labels.shape[0])
    #     np.random.shuffle(idxs)
    #     train_idx, val_idx, test_idx = idxs[val_num + test_num:], idxs[:val_num], idxs[val_num:val_num + test_num]
    #     g = dgl.node_subgraph(g, {args.label_entity: labels[:, 0], })
    #     user_features = user_features[labels[:, 0]]
    #     device_features = device_features[g.nodes['device'].data[dgl.NID]]
    #     labels = torch.tensor(labels[:, 1], dtype=torch.int64, device=device)
    # else:
    np.random.shuffle(labels)
    train_idx, val_idx, test_idx = \
        labels[val_num + test_num:, 0], labels[:val_num, 0], labels[val_num:val_num + test_num, 0]
    expand_labels = np.empty(user_features.shape[0], dtype=np.int64)
    expand_labels[labels[:, 0]] = labels[:, 1]
    labels = torch.tensor(expand_labels, device=device)
    #
    # user_features = F.pad(torch.tensor(user_features, device=device, dtype=torch.float32), (0, num_device_feature))
    # device_features = F.pad(torch.tensor(device_features, device=device, dtype=torch.float32), (num_user_feature, 0))
    user_features = torch.tensor(user_features, device=device, dtype=torch.float32)
    device_features = torch.tensor(device_features, device=device, dtype=torch.float32)

    entity_features = {'user': user_features, 'device': device_features}

    # g.edges['used'].data['weight'] = torch.ShortTensor(relation_df['relation_edge_weight'].values)
    # g.edges['used-by'].data['weight'] = torch.ShortTensor(relation_df['relation_edge_weight'].values)
    # del relation_df
    # gc.collect()

    # prepare for training

    data = train_idx, val_idx, test_idx, num_user_feature + num_device_feature, num_user_feature, num_device_feature, \
           labels, n_classes, entity_features, g

    # Run 10 times
    test_accs = []
    for i in range(10):
        test_accs.append(train(args, device, data))
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=thread_wrapped_func(run),
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()