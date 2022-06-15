from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, remove_self_loops
import math

class GLAPool(torch.nn.Module):
    def __init__(self, in_channels, alpha, ratio=0, non_linearity=torch.tanh):
        super(GLAPool, self).__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.ratio = ratio
        self.non_linearity = non_linearity
        self.score1 = nn.Linear(self.in_channels, 1)
        self.score2 = GCNConv(in_channels=self.in_channels, out_channels=1, add_self_loops=False)

    def reset_parameters(self):
        self.score1.reset_parameters()
        self.score2.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, flag=0):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_index1, _ = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)
        score = (self.alpha * self.score1(x) + (1-self.alpha) * self.score2(x, edge_index1)).squeeze()

        if flag == 1:
            return score.view(-1,1)
        else:
            perm = topk(score, self.ratio, batch)
            x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(
                edge_index, edge_attr, perm, num_nodes=score.size(0))

            return x, edge_index, batch


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.softmax_dim = 2

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = GCNConv(dim_K, dim_V)
        self.fc_v = GCNConv(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.fc_k.reset_parameters()
        self.fc_v.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        self.fc_o.reset_parameters()

    def forward(self, Q, graph=None):

        Q = self.fc_q(Q)

        (x, edge_index, batch) = graph
        K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)
        K, mask = to_dense_batch(K, batch)
        V, _ = to_dense_batch(V, batch)
        attention_mask = mask.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -1e9

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), 0)
        K_ = torch.cat(K.split(dim_split, dim=2), 0)
        V_ = torch.cat(V.split(dim_split, dim=2), 0)

        attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
        attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = torch.softmax(attention_mask + attention_score, self.softmax_dim)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)

        return O



