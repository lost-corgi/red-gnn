import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import *
# from eval_utils import *
from model import binaryRGCN
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score

def train(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, input_dim, left_pad_size, right_pad_size, labels, n_classes, entity_features, g = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        {args.label_entity: train_nid},
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)   # may change device to gpu? default device='cpu'

    # Define model and optimizer
    model = binaryRGCN(input_dim, args.hidden_dim, args.num_layers, F.relu, args.dropout, g.etypes, args.label_entity)
    model = model.to(device)
    loss_fcn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_auc = 0
    best_test_auc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(entity_features, labels, seeds, input_nodes, args.label_entity, args.is_pad, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs) # pred
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                auc = roc_auc_score(batch_labels.cpu().detach().numpy(), batch_pred.detach().numpy())
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train auc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), auc, np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                pred = model.inference(g, entity_features, device, args.batch_size, args.num_workers, args.is_pad)
            model.train()
            eval_auc = roc_auc_score(labels[val_nid].cpu().detach().numpy(), pred[val_nid].detach().numpy())
            test_auc = roc_auc_score(labels[test_nid].cpu().detach().numpy(), pred[test_nid].detach().numpy())

            # if args.save_pred:
            #     np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            print('Eval auc {:.4f}'.format(eval_auc))
            if eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                best_test_auc = test_auc
            print('Best Eval auc {:.4f} Test auc {:.4f}'.format(best_eval_auc, best_test_auc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    return best_test_auc


# def run(proc_id, n_gpus, args, devices, data):
#     # Start up distributed training, if enabled.
#     dev_id = devices[proc_id]
#     if n_gpus > 1:
#         dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
#             master_ip='127.0.0.1', master_port='12345')
#         world_size = n_gpus
#         th.distributed.init_process_group(backend="nccl",
#                                           init_method=dist_init_method,
#                                           world_size=world_size,
#                                           rank=proc_id)
#     th.cuda.set_device(dev_id)
#
#     # Unpack data
#     n_classes, train_g, val_g, test_g = data
#
#     if args.inductive:
#         train_nfeat = train_g.ndata.pop('features')
#         val_nfeat = val_g.ndata.pop('features')
#         test_nfeat = test_g.ndata.pop('features')
#         train_labels = train_g.ndata.pop('labels')
#         val_labels = val_g.ndata.pop('labels')
#         test_labels = test_g.ndata.pop('labels')
#     else:
#         train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
#         train_labels = val_labels = test_labels = g.ndata.pop('labels')
#
#     if not args.data_cpu:
#         train_nfeat = train_nfeat.to(dev_id)
#         train_labels = train_labels.to(dev_id)
#
#     in_feats = train_nfeat.shape[1]
#
#     train_mask = train_g.ndata['train_mask']
#     val_mask = val_g.ndata['val_mask']
#     test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
#     train_nid = train_mask.nonzero().squeeze()
#     val_nid = val_mask.nonzero().squeeze()
#     test_nid = test_mask.nonzero().squeeze()
#
#     # Split train_nid
#     train_nid = th.split(train_nid, math.ceil(len(train_nid) / n_gpus))[proc_id]
#
#     # Create PyTorch DataLoader for constructing blocks
#     sampler = dgl.dataloading.MultiLayerNeighborSampler(
#         [int(fanout) for fanout in args.fan_out.split(',')])
#     dataloader = dgl.dataloading.NodeDataLoader(
#         train_g,
#         train_nid,
#         sampler,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=False,
#         num_workers=args.num_workers)
#
#     # Define model and optimizer
#     model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
#     model = model.to(dev_id)
#     if n_gpus > 1:
#         model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
#     loss_fcn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     # Training loop
#     avg = 0
#     iter_tput = []
#     for epoch in range(args.num_epochs):
#         tic = time.time()
#
#         # Loop over the dataloader to sample the computation dependency graph as a list of
#         # blocks.
#         for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
#             if proc_id == 0:
#                 tic_step = time.time()
#
#             # Load the input features as well as output labels
#             batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
#                                                         seeds, input_nodes, dev_id)
#             blocks = [block.int().to(dev_id) for block in blocks]
#             # Compute loss and prediction
#             batch_pred = model(blocks, batch_inputs)
#             loss = loss_fcn(batch_pred, batch_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if proc_id == 0:
#                 iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
#             if step % args.log_every == 0 and proc_id == 0:
#                 acc = compute_acc(batch_pred, batch_labels)
#                 print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
#                     epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))
#
#         if n_gpus > 1:
#             th.distributed.barrier()
#
#         toc = time.time()
#         if proc_id == 0:
#             print('Epoch Time(s): {:.4f}'.format(toc - tic))
#             if epoch >= 5:
#                 avg += toc - tic
#             if epoch % args.eval_every == 0 and epoch != 0:
#                 if n_gpus == 1:
#                     eval_acc = evaluate(
#                         model, val_g, val_nfeat, val_labels, val_nid, devices[0])
#                     test_acc = evaluate(
#                         model, test_g, test_nfeat, test_labels, test_nid, devices[0])
#                 else:
#                     eval_acc = evaluate(
#                         model.module, val_g, val_nfeat, val_labels, val_nid, devices[0])
#                     test_acc = evaluate(
#                         model.module, test_g, test_nfeat, test_labels, test_nid, devices[0])
#                 print('Eval Acc {:.4f}'.format(eval_acc))
#                 print('Test Acc: {:.4f}'.format(test_acc))
#
#
#     if n_gpus > 1:
#         th.distributed.barrier()
#     if proc_id == 0:
#         print('Avg epoch time: {}'.format(avg / (epoch - 4)))