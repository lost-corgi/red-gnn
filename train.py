import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from data_utils import *
# from eval_utils import *
from model import binaryRGCN
import dgl
import numpy as np
import math
from sklearn.metrics import roc_auc_score


def train(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, input_dim, _, _, labels, n_classes, entity_features, g = data

    # Create PyTorch DataLoader for constructing blocks
    train_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        {args.label_entity: train_nid},
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)  # may change device to gpu? default device='cpu'

    val_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        {args.label_entity: val_nid},
        val_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = binaryRGCN(input_dim, args.hidden_dim, args.num_layers, F.relu, args.dropout, g.etypes, args.label_entity)
    model = model.to(device)
    loss_fcn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Train/Val/Test
    avg = 0
    iter_tput = []
    best_val_auc = 0
    best_model_state_dict = model.state_dict()
    for epoch in range(args.num_epochs):
        tic = time.time()
        # train prcess: loop over the dataloader to sample the computation dependency graph as a list of blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            tic_step = time.time()
            blocks = [blk.int().to(device) for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(entity_features, labels, seeds, input_nodes, args.label_entity,
                                                        args.is_pad, device)
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                auc = roc_auc_score(batch_labels.cpu().detach().numpy(), batch_pred.cpu().detach().numpy())
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:ls.4f} | Train auc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, loss.item(), auc, np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic

        # valid process
        if epoch % args.eval_every == 0 and epoch != 0:
            val_loss = 0
            count = 0
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for input_nodes, seeds, blocks in val_dataloader:
                    pred_size = len(seeds[args.label_entity])
                    blocks = [blk.int().to(device) for blk in blocks]
                    batch_inputs, batch_labels = load_subtensor(entity_features, labels, seeds, input_nodes,
                                                                args.label_entity, args.is_pad, device)
                    batch_pred = model(blocks, batch_inputs)
                    loss = loss_fcn(batch_pred, batch_labels)
                    val_loss += loss.item() * pred_size
                    val_preds.append(batch_pred.squeeze().cpu().detach().numpy())
                    val_labels.append(batch_labels.squeeze().cpu().detach().numpy())
                    count += pred_size

                val_loss /= count
                val_preds = np.concatenate(val_preds)
                val_labels = np.concatenate(val_labels)
                val_auc = roc_auc_score(val_labels, val_preds)
                print("Epoch {:05d} | Valid loss: {:.4f} | Valid Auc: {:.4f}".format(epoch, val_loss, val_auc))
            model.train()
            # if args.save_pred:
            #     np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state_dict = model.state_dict()
            print('Best val auc {:.4f}'.format(best_val_auc))
            # print(model.state_dict())
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    # test process: retrieve best validation model for test evaluation
    best_model = binaryRGCN(input_dim, args.hidden_dim, args.num_layers, F.relu, args.dropout, g.etypes,
                            args.label_entity)
    best_model.load_state_dict(best_model_state_dict)
    best_model.eval()
    with torch.no_grad():
        pred = best_model.inference(g, entity_features, device, args.batch_size, args.num_workers, args.is_pad)
    test_auc = roc_auc_score(labels[test_nid].squeeze().cpu().detach().numpy(),
                             pred[test_nid].squeeze().detach().numpy())
    # np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.4f}'.format})
    # print(labels[test_nid].squeeze().cpu().detach().numpy())
    # print(pred[test_nid].squeeze().detach().numpy())
    print('Test set auc {:.4f}'.format(test_auc))
    return test_auc


def train_mp(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=proc_id)
    torch.cuda.set_device(dev_id)

    # Unpack data
    train_nid, val_nid, test_nid, input_dim, _, _, labels, n_classes, entity_features, g = data

    if not args.data_cpu:
        entity_features = {k: v.to(dev_id) for k, v in entity_features.items()}

    # Split train/val_nid
    train_nid = torch.split(train_nid, math.ceil(len(train_nid) / n_gpus))[proc_id]
    # val_nid = torch.split(val_nid, math.ceil(len(val_nid) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    train_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        {args.label_entity: train_nid},
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)  # may change device to gpu? default device='cpu'

    val_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        {args.label_entity: val_nid},
        val_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = binaryRGCN(input_dim, args.hidden_dim, args.num_layers, F.relu, args.dropout, g.etypes, args.label_entity)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Train/Val/Test
    avg = 0
    iter_tput = []
    best_val_auc = 0
    best_model_state_dict = model.state_dict()
    for epoch in range(args.num_epochs):
        tic = time.time()
        # train prcess: loop over the dataloader to sample the computation dependency graph as a list of blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            if proc_id == 0:
                tic_step = time.time()
            blocks = [blk.int().to(dev_id) for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(entity_features, labels, seeds, input_nodes, args.label_entity,
                                                        args.is_pad, dev_id)
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if proc_id == 0:
                iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
            if step % args.log_every == 0 and proc_id == 0:
                auc = roc_auc_score(batch_labels.cpu().detach().numpy(), batch_pred.cpu().detach().numpy())
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:ls.4f} | Train auc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, loss.item(), auc, np.mean(iter_tput[3:]), torch.cuda.max_memory_allocated() / 1000000))

        if n_gpus > 1:
            torch.distributed.barrier()
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            # valid process
            if epoch % args.eval_every == 0 and epoch != 0:
                if n_gpus > 1:
                    model = model.module
                val_loss = 0
                count = 0
                model.eval()
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for input_nodes, seeds, blocks in val_dataloader:
                        pred_size = len(seeds[args.label_entity])
                        blocks = [blk.int().to(dev_id) for blk in blocks]
                        batch_inputs, batch_labels = load_subtensor(entity_features, labels, seeds, input_nodes,
                                                                    args.label_entity, args.is_pad, dev_id)
                        batch_pred = model(blocks, batch_inputs)
                        loss = loss_fcn(batch_pred, batch_labels)
                        val_loss += loss.item() * pred_size
                        val_preds.append(batch_pred.squeeze().cpu().detach().numpy())
                        val_labels.append(batch_labels.squeeze().cpu().detach().numpy())
                        count += pred_size

                    val_loss /= count
                    val_preds = np.concatenate(val_preds)
                    val_labels = np.concatenate(val_labels)
                    val_auc = roc_auc_score(val_labels, val_preds)
                    print("Epoch {:05d} | Valid loss: {:.4f} | Valid Auc: {:.4f}".format(epoch, val_loss, val_auc))
                model.train()
                # if args.save_pred:
                #     np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state_dict = model.state_dict()
                print('Best val auc {:.4f}'.format(best_val_auc))
                # print(model.state_dict())

    if n_gpus > 1:
        torch.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))
        # test process: retrieve best validation model for test evaluation
        best_model = binaryRGCN(input_dim, args.hidden_dim, args.num_layers, F.relu, args.dropout, g.etypes,
                                args.label_entity)
        best_model.load_state_dict(best_model_state_dict)
        best_model.eval()
        with torch.no_grad():
            pred = best_model.inference(g, entity_features, dev_id, args.batch_size, args.num_workers, args.is_pad)
        test_auc = roc_auc_score(labels[test_nid].squeeze().cpu().detach().numpy(),
                                 pred[test_nid].squeeze().detach().numpy())
        # np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.4f}'.format})
        # print(labels[test_nid].squeeze().cpu().detach().numpy())
        # print(pred[test_nid].squeeze().detach().numpy())
        print('Test set auc {:.4f}'.format(test_auc))