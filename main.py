#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import ssl

import copy
import itertools
import random
import torch
import numpy as np
from utils.options import args_parser
from utils.seed import setup_seed
from utils.logg import get_logger
from models.Nets import client_model
from models.Nets_VIB import client_model_VIB
from utils.utils_dataset import DatasetObject
from models.distributed_training_utils import Client, Server
torch.set_printoptions(
    precision=8,
    threshold=1000,
    edgeitems=3,
    linewidth=150,
    profile=None,
    sci_mode=False
)


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    # parse args
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    data_path = 'Folder/'
    data_obj = DatasetObject(dataset=args.dataset, n_client=args.num_users, seed=args.seed, rule=args.rule, class_main=args.class_main, data_path=data_path, frac_data=args.frac_data, dir_alpha=args.dir_a)

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y;
    tst_x = data_obj.tst_x;
    tst_y = data_obj.tst_y

    # build model
    if args.method == 'FedCR':
        net_glob = client_model_VIB(args, args.dimZ, args.alpha, args.dataset).to(args.device)
    else:
        if args.dataset == 'CIFAR100':
            net_glob = client_model('cifar100_LeNet').to(args.device)
        elif args.dataset == 'CIFAR10':
            net_glob = client_model('cifar10_LeNet').to(args.device)
        elif args.dataset == 'EMNIST':
            net_glob = client_model('emnist_NN', [1 * 28 * 28, 10]).to(args.device)
        elif args.dataset == 'FMNIST':
            net_glob = client_model('FMNIST_CNN', [1 * 28 * 28, 10]).to(args.device)
        else:
            exit('Error: unrecognized model')

    # Set the annotation for CIFAR10 by default
    total_num_layers = len(net_glob.state_dict().keys())  # 10
    # convert order_keys to dict net keys
    # [['conv1.weight', 'conv1.bias'],
    # ['conv2.weight', 'conv2.bias'],
    # ['fc1.weight', 'fc1.bias'],
    # ['fc2.weight', 'fc2.bias'],
    # ['fc3.weight', 'fc3.bias']]
    net_keys = [*net_glob.state_dict().keys()]

    if (args.method == 'fedrep' or args.method == 'fedper' or args.method == 'fedbabu' or args.method == 'FLIX'
            or args.method == 'Scafflix' or args.method == 'FedAvg2'):
        # FedAvg2: client_side model for evaluation
        # FedAvg: server_side model for evaluation
        if 'CIFAR100' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1]]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
        else:
            exit('Error: unrecognized data1')
    elif args.method == 'lg':
        if 'CIFAR100' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3, 4]]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3, 4]]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3, 4]]
        else:
            exit('Error: unrecognized data2')
    elif args.method in ['LowerB', 'OPU2', 'OPU3']:
        if 'CIFAR100' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
            w_final_keys = net_glob.weight_keys[4]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
            w_final_keys = net_glob.weight_keys[4]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1]]
            w_final_keys = net_glob.weight_keys[2]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
            w_final_keys = net_glob.weight_keys[4]
        else:
            exit('Error: unrecognized data1')
    elif args.method == 'FedCR':
        if 'CIFAR100' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3, 4]]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3, 4]]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3, 4]]
        else:
            exit('Error: unrecognized data3')
    elif args.method == 'fedavg' or args.method == 'ditto' or args.method == 'maml':
        w_glob_keys = []
    else:
        exit('Error: unrecognized data4')

    # [['conv1.weight', 'conv1.bias'], ['conv2.weight', 'conv2.bias'], ['fc1.weight', 'fc1.bias'], ['fc2.weight', 'fc2.bias']]
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    # ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']

    clients = [Client(model=copy.deepcopy(net_glob).to(args.device), args=args, trn_x=data_obj.clnt_x[i],
                      trn_y=data_obj.clnt_y[i], tst_x=data_obj.tst_x[i], tst_y=data_obj.tst_y[i],
                      n_cls = data_obj.n_cls, dataset_name=data_obj.dataset, id_num=i) for i in range(args.num_users)]
    server = Server(model = (net_glob).to(args.device), args = args, n_cls = data_obj.n_cls)

    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')
    logger.info('total_num_layers')
    logger.info(total_num_layers)
    logger.info('net_keys')
    logger.info(net_keys)
    logger.info('w_glob_keys')
    logger.info(w_glob_keys)

    logger.info('start training!')

    for iter in range(args.epochs + 1):
        net_glob.train()

        # m = max(int(args.frac * args.num_users), 1)
        # TODO: Allow partial participation
        m = max(int(args.num_users), 1)  # we always consider full participation
        if iter == args.epochs:
            m = args.num_users
        participating_clients = random.sample(clients, m)

        last = iter == args.epochs
        w_glob_keys_ops = {}
        for client in participating_clients:
            # convey the client id for each update, customize the layers for training then
            if args.method == 'LowerB':
                if args.dataset in ['CIFAR10', 'CIFAR100', 'FMNIST']:
                    w_glob_keys_op = net_glob.weight_keys[int(client.id/25)]
                elif args.dataset == 'EMNIST':
                    w_glob_keys_op = net_glob.weight_keys[int(client.id/50)]
            elif args.method == 'OPU2': # Overlapped uniformed 2
                if args.dataset in ['CIFAR10', 'CIFAR100', 'FMNIST']:
                    d1 = int(client.id/25)
                    d2 = (int(client.id/25)+1) % int(total_num_layers/2-1)
                    w_glob_keys_op = [net_glob.weight_keys[i] for i in [d1, d2]]
                else:
                    w_glob_keys_op = w_glob_keys
            elif args.method == 'OPU3':
                if args.dataset in ['CIFAR10', 'CIFAR100', 'FMNIST']:
                    d1 = int(client.id/25)
                    d2 = (int(client.id/25)+1) % int(total_num_layers/2-1)
                    d3 = (int(client.id/25)+2) % int(total_num_layers/2-1)
                    w_glob_keys_op = [net_glob.weight_keys[i] for i in [d1, d2, d3]]
                else:
                    w_glob_keys_op = w_glob_keys
            elif args.method == 'OPR2': # Overlapped random 2
                if args.dataset in ['CIFAR10', 'CIFAR100', 'FMNIST']:
                    ds = np.random.choice(int(total_num_layers/2-1), 2, replace=False)
                    w_glob_keys_op = [net_glob.weight_keys[i] for i in ds]
                else:
                    w_glob_keys_op = w_glob_keys
            else:
                w_glob_keys_op = w_glob_keys

            w_glob_keys_op = list(itertools.chain.from_iterable(w_glob_keys_op))
            w_glob_keys_op = w_glob_keys_op + w_final_keys
            w_glob_keys_ops[client.id] = w_glob_keys_op
            # print(w_glob_keys_op)

            if args.sync == 'True':
                if args.method in ['LowerB', 'OPU2', 'OPU3']:
                    client.synchronize_with_server(server, w_glob_keys, args.global_prune, w_glob_keys_op)
                else:
                    client.synchronize_with_server(server, w_glob_keys)

            client.compute_weight_update(w_glob_keys, server, last)

        if args.method in ['LowerB', 'OPU2', 'OPU3']:
            server.aggregate_weight_updates_subset(clients=participating_clients, iter=iter,
                                                   w_glob_keys_ops=w_glob_keys_ops, aggregation=args.aggregation)
        else:
            server.aggregate_weight_updates(clients=participating_clients, iter=iter)
        print('\n')

        if args.method == 'FedCR':
            server.global_POE(clients=participating_clients)


#-----------------------------------------------test--------------------------------------------------------------------

#-----------------------------------------------test--------------------------------------------------------------------
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            results_loss =[]; results_acc = []
            results_loss_last = []; results_acc_last = []
            for client in clients:
                if args.method == 'FedCR':
                    results_test, loss_test1 = client.evaluate_FedVIB(data_x=client.tst_x, data_y=client.tst_y,
                                                               dataset_name=data_obj.dataset)
                elif args.method == 'FLIX' or args.method == 'Scafflix':
                    results_test, loss_test1 = client.evaluate_FLIX(data_x=client.tst_x, data_y=client.tst_y,dataset_name=data_obj.dataset, server=server)
                elif args.method != 'fedavg':  # Decision for FedAvg2 as well
                    results_test, loss_test1 = client.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                               dataset_name=data_obj.dataset)
                elif args.method == 'fedavg':
                    results_test, loss_test1 = server.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                               dataset_name=data_obj.dataset)
                    if last:
                        results_test_last, loss_test1_last = client.evaluate(data_x=client.tst_x, data_y=client.tst_y, dataset_name=data_obj.dataset)
                results_loss.append(loss_test1)
                results_acc.append(results_test)

                if last and args.method == 'fedavg':
                    results_loss_last.append(loss_test1_last)
                    results_acc_last.append(results_test_last)

            results_loss = np.mean(results_loss)
            results_acc = np.mean(results_acc)
            if last:
                logger.info('Final Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(iter, args.lr, results_loss, results_acc))
            else:
                logger.info('Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(iter, args.lr, results_loss, results_acc))

            if last and args.method == 'fedavg':
                results_loss_last= np.mean(results_loss_last)
                results_acc_last= np.mean(results_acc_last)
                logger.info('Final FT Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(iter, args.lr, results_loss_last, results_acc_last))

        args.lr = args.lr * (args.lr_decay)

    logger.info('finish training!')






