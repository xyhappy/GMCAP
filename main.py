import argparse
import os
import random
import torch
import numpy as np
from math import ceil
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from network import Net
from train_eval import cross_validation
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=int, default=0.9, help='alpha')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--dataname', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--dataset_path', type=str, default='../pyg_data', help='path to save result')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=3, help='maximum number of epochs')
parser.add_argument('--experiment_number', default='1', type=str)
parser.add_argument('--folds', type=int, default=10, help='Cross validation folds')
parser.add_argument('--kernels', type=int, default=2, help='kernels of each gconv layer')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--num_heads', type=int, default=1, help='attention head size')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--save_path', type=str, default='results', help='path to save results')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')


args = parser.parse_args()
if torch.cuda.is_available():
    if args.dataname == 'PROTEINS':
        args.device = 'cuda:0'
    elif args.dataname == 'DD':
        args.device = 'cuda:0'
    elif args.dataname == 'NCI1':
        args.device = 'cuda:0'
    elif args.dataname == 'NCI109':
        args.device = 'cuda:1'
    elif args.dataname == 'Mutagenicity':
        args.device = 'cuda:0'
    elif args.dataname == 'COLLAB':
        args.device = 'cuda:1'
else:
    args.device = 'cpu'

def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

if __name__ == "__main__":

    dataset = TUDataset(os.path.join(args.dataset_path, args.dataname), name=args.dataname)
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    args.max_num_nodes = max([g.num_nodes for g in dataset])
    args.mean_num_nodes = ceil(np.mean([g.num_nodes for g in dataset]))

    # data_path
    if not os.path.isdir(args.dataset_path):
        os.makedirs(args.dataset_path)

    # results
    args.results = './{}/{}'.format(args.save_path, args.dataname)
    if not os.path.isdir(args.results):
        os.makedirs(args.results)

    setseed(args.seed)
    model = Net(args)
    acc, std, duration_mean, overtime = cross_validation(
        dataset,
        model,
        args
    )


