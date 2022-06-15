import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from layer import GLAPool, MAB
from math import ceil

class MKGC(torch.nn.Module):
    def __init__(self, kernels, in_channel, out_channel):
        super(MKGC, self).__init__()
        self.kernels = kernels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g_list = torch.nn.ModuleList()
        for i in range(self.kernels):
            self.g_list.append(GCNConv(in_channel, out_channel))

    def reset_parameters(self):
        for gconv in self.g_list:
            gconv.reset_parameters()

    def forward(self, x, edge_index):
        total_x = None
        for gconv in self.g_list:
            feature = gconv(x, edge_index)
            if total_x == None:
                total_x = F.relu(feature)
            else:
                total_x = total_x + F.relu(feature)
        return total_x


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.alpha = args.alpha
        self.kernels = args.kernels
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_heads = args.num_heads
        self.mean_num_nodes = args.mean_num_nodes
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.ratio = ceil(int(self.mean_num_nodes * self.pooling_ratio))

        self.node_encoder = torch.nn.Linear(self.num_features, self.nhid)
        self.gconv1 = MKGC(self.kernels, self.nhid, self.nhid)
        self.gconv2 = MKGC(self.kernels, self.nhid, self.nhid)
        self.gconv3 = MKGC(self.kernels, self.nhid, self.nhid)
        self.weight = GLAPool(self.nhid, self.alpha)
        self.pool_att = Pool_Att(self.nhid, self.alpha, self.ratio, self.num_heads)
        self.classifier = Classifier(self.nhid, self.dropout_ratio, self.num_classes)

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.gconv1.reset_parameters()
        self.gconv2.reset_parameters()
        self.gconv3.reset_parameters()
        self.weight.reset_parameters()
        self.pool_att.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x1 = self.gconv1(x, edge_index)
        x1 = F.dropout(x1, p=self.dropout_ratio, training=self.training)
        x2 = self.gconv2(x1, edge_index)
        x2 = F.dropout(x2, p=self.dropout_ratio, training=self.training)
        x3 = self.gconv3(x2, edge_index)

        weight = torch.cat((self.weight(x1, edge_index, None, batch, 1),
                            self.weight(x2, edge_index, None, batch, 1),
                            self.weight(x3, edge_index, None, batch, 1)), dim=-1)
        weight = torch.softmax(weight, dim=-1)
        x = weight[:, 0].view(-1, 1) * x1 + weight[:, 1].view(-1, 1) * x2 + weight[:, 2].view(-1, 1) * x3

        x = self.pool_att(x, edge_index, batch)
        graph_feature = x
        x = self.classifier(x)

        return x, graph_feature

class Pool_Att(torch.nn.Module):
    def __init__(self, nhid, alpha, ratio, num_heads):
        super(Pool_Att, self).__init__()
        self.ratio = ratio
        self.pool = GLAPool(nhid, alpha, self.ratio)
        self.att = MAB(nhid, nhid, nhid, num_heads)
        self.readout = torch.nn.Conv1d(self.ratio, 1, 1)

    def reset_parameters(self):
        self.pool.reset_parameters()
        self.att.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, x, edge_index, batch):
        graph = (x, edge_index, batch)
        xp, _, batchp = self.pool(x=x, edge_index=edge_index, batch=batch) # 从n个节点选出k个节点
        xp, _ = to_dense_batch(x=xp, batch=batchp, max_num_nodes=self.ratio, fill_value=0)
        xp = self.att(xp, graph)
        xp = self.readout(xp).squeeze()
        return xp


class Classifier(torch.nn.Module):
    def __init__(self, nhid, dropout_ratio, num_classes):
        super(Classifier, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.lin1 = torch.nn.Linear(nhid, nhid)
        self.lin2 = torch.nn.Linear(nhid, num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x



