from argparse import ArgumentParser
import numpy as np
import torch
from data import get_dataset, DATASET_CONFIGS
from train import train
from model import MLP
import utils

parser = ArgumentParser('EWC Implementation')
parser.add_argument('-hidden_size', type=int, default=400)
parser.add_argument('-hidden_layer_num', type=int, default=2)
parser.add_argument('-hidden_dropout_prob',type=float, default=0.5)
parser.add_argument('-input_dropout_prob',type=float, default=0.2)
parser.add_argument('-task_number', type=int, default=8)
parser.add_argument('-epochs_per_task', type=int, default=3)
parser.add_argument('-lamda', type=float, default=40)
parser.add_argument('-lr', type=float, default=1e-1)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-weight_decay', type=float, default=0)
parser.add_argument('-test_size', type=int, default=1024)
parser.add_argument('-random_seed', type=int, default=0)
parser.add_argument('-no_gpu', action='store_false', dest='cuda')
parser.add_argument('-fisher_estm_sample_size', type=int, default=1024)
parser.add_argument('-eval_log_interval', type=int, default=250)
parser.add_argument('-loss_log_interval', type=int, default=250)
parser.add_argument('-consolidate_', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    cuda = torch.cuda.is_available() and args.cuda

    np.random.seed(args.random_seed)
    permutations = [
        np.random.permutation(DATASET_CONFIGS['mnist']['size']**2) for _ in range(args.task_number)
    ]

    train_dataset = [
        get_dataset('mnist', permutation=p) for p in permutations
    ]

    mlp = MLP(
        DATASET_CONFIGS['mnist']['size']**2,
        DATASET_CONFIGS['mnist']['classes'],
        hidden_size=args.hidden_size,
        hidden_layer_num=args.hidden_layer_num,
        hidden_dropout_prob=args.hidden_dropout_prob,
        input_dropout_prob=args.input_dropout_prob,
        lamda=args.lamda,
    )

    utils.xavier_initialize(mlp)

    if cuda:
        mlp.cuda()

    train(
        mlp, train_datasets, test_datasets,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        test_size=args.test_size,
        consolidate=args.consolidate,
        fisher_estimation_sample_size=args.fisher_estimation_sample_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_log_interval=args.eval_log_interval,
        loss_log_interval=args.loss_log_interval,
        cuda=cuda
    )


    