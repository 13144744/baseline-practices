import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hgcn.manifolds.poincare import PoincareBall
from sparse_softmax import Sparsemax
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce


class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            # data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min', fill_value=fill)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min')
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)
        self.manifold = PoincareBall()
        self.c = 1.0
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class HgpslPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2):
        super(HgpslPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x_tan = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        x_information_score = self.calc_information_score(x_tan, edge_index, edge_attr)  # 对每个节点评分
        score = torch.sum(torch.abs(x_information_score), dim=1)  # 评分绝对值求和

        # Graph Pooling
        original_x = x
        perm = topk(score, self.ratio, batch)  # topK获取保留节点的编号
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # Discard structure learning layer, directly return
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0))

            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            # weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            # weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb
            att_tan = self.manifold.proj_tan0(self.manifold.logmap0(self.att, c=self.c), c=self.c)
            weights = self.manifold.mobius_matvec(att_tan, torch.cat([x[row], x[col]], dim=1), self.c)
            lamb_new_edge_attr = self.manifold.mobius_matvec(self.lamb, new_edge_attr, self.c)
            weights = self.manifold.mobius_add(F.leaky_relu(weights, self.negative_slop), lamb_new_edge_attr, self.c)
            weights = self.manifold.proj_tan0(self.manifold.logmap0(weights, c=self.c), c=self.c)
            del att_tan, lamb_new_edge_attr

            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            # del adj
            torch.cuda.empty_cache()
        else:
            # Learning the possible edge weights between each pair of nodes in the pooled subgraph, relative slower.
            if edge_attr is None:
                induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
                                               device=induced_edge_index.device)
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            # Construct batch fully connected graph in block diagonal matirx format
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index

            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index

            adj[induced_row, induced_col] += induced_edge_attr * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            # del adj
            torch.cuda.empty_cache()

        return x, new_edge_index, new_edge_attr, batch, adj


class HypbolicPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2):
        super(HypbolicPool, self).__init__()
        self.manifold = PoincareBall()
        self.c = 1.0

        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_tan = self.manifold.logmap0(x, c=self.c)
        x_information_score = self.calc_information_score(x_tan, edge_index, edge_attr)  # 对每个节点评分
        score = torch.sum(torch.abs(x_information_score), dim=1)  # 评分绝对值求和

        # Graph Pooling
        original_x = x
        perm = topk(score, self.ratio, batch)  # topK获取保留节点的编号
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # Discard structure learning layer, directly return
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0))
            del hop_edge_index, hop_edge_attr

            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            # 切空间做SL
            # weights = (torch.cat([x_tan[row], x_tan[col]], dim=1) * self.att).sum(dim=-1)
            # weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb

            weights = torch.cat([self.manifold.logmap0(x[row], self.c), self.manifold.logmap0(x[col], self.c)], dim=1)
            weights = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(weights, self.c), self.c),
                                         self.c)
            att = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(self.att, self.c), self.c), self.c)

            # mobius_matvec
            weights = self.manifold.mobius_matvec(att, weights, self.c).sum(dim=-1)
            weights = self.manifold.proj(weights, self.c)

            hyp_b = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(new_edge_attr * self.lamb, self.c), self.c), self.c)
            # mobius_add
            weights = self.manifold.mobius_add(weights, hyp_b, c=self.c)
            weights = self.manifold.proj(weights, self.c)
            weights = self.manifold.proj_tan0(self.manifold.logmap0(weights, c=self.c), c=self.c)

            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del weights

            # del adj
            torch.cuda.empty_cache()
        # else:
        #     # Learning the possible edge weights between each pair of nodes in the pooled subgraph, relative slower.
        #     if edge_attr is None:
        #         induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
        #                                        device=induced_edge_index.device)
        #     num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        #     shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
        #     cum_num_nodes = num_nodes.cumsum(dim=0)
        #     adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
        #     # Construct batch fully connected graph in block diagonal matirx format
        #     for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
        #         adj[idx_i:idx_j, idx_i:idx_j] = 1.0
        #     new_edge_index, _ = dense_to_sparse(adj)
        #     row, col = new_edge_index
        #
        #     weights = (torch.cat([x_tan[row], x_tan[col]], dim=1) * self.att).sum(dim=-1)
        #     weights = F.leaky_relu(weights, self.negative_slop)
        #     adj[row, col] = weights
        #     induced_row, induced_col = induced_edge_index
        #
        #     adj[induced_row, induced_col] += induced_edge_attr * self.lamb
        #     weights = adj[row, col]
        #     if self.sparse:
        #         new_edge_attr = self.sparse_attention(weights, row)
        #     else:
        #         new_edge_attr = softmax(weights, row, x.size(0))
        #     # filter out zero weight edges
        #     adj[row, col] = new_edge_attr
        #     new_edge_index, new_edge_attr = dense_to_sparse(adj)
        #     # release gpu memory
        #     # del adj
        #     torch.cuda.empty_cache()

        return x, new_edge_index, new_edge_attr, batch, adj


class NewPool(torch.nn.Module):
    def __init__(self, args, feat_in, feat_out, spread=1, bias=False):
        super(NewPool, self).__init__()
        self.manifold = PoincareBall()
        self.c = args.c
        self.L_flag = torch.zeros(1, 1).cuda()
        self.bias = bias
        self.ratio = 0.8
        self.updater = nn.Linear(feat_in, feat_out, bias=self.bias)
        self.p_leader = nn.Linear(feat_out, 2, bias=self.bias)
        self.layer_weight = nn.Linear(2 * feat_out, 1, bias=self.bias)
        self.aggregator = PROPAGATION_OUT()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, T=1):
        x_tan = self.manifold.logmap0(x, c=self.c)
        edge_attr = None
        x_information_score = self.calc_information_score(x_tan, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        # random_prob = F.softmax(score, dim=-1)

        # self.prob_i = random_prob.unsqueeze(1)


        # PRELIMINARY CALCULATIONS
        self.L_flag = torch.zeros(1, 1).cuda()

        self.L_flag = self.L_flag * 0
        updated_x = F.leaky_relu(self.updater(x_tan))
        sum_Neigh_x = self.aggregator(updated_x, edge_index)

        #  SELECTION  <==============================================================
        # random_prob = F.relu(self.p_leader(sum_Neigh_x))
        # random_prob = F.softmax(random_prob, dim=-1)
        #
        # self.prob_i = random_prob[:, 1].unsqueeze(1)
        values, indices = score.topk(int(len(score) * self.ratio), dim=0, largest=True, sorted=True)
        T = torch.min(values)
        hot_prob = torch.where(score > 1, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        SEL_v = hot_prob.view(-1, 1)
        self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        sum_SEL_x = self.aggregator(SEL_v * updated_x, edge_index)
        concat_sums = torch.cat([sum_SEL_x, sum_Neigh_x], dim=-1)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        A_x = F.relu(self.aggregator(weight_SEL_v * SEL_v * updated_x, edge_index))

        out = updated_x + A_x
        out = self.manifold.proj(self.manifold.expmap0(out, self.c), self.c)
        return out


class NewPoolTb(torch.nn.Module):
    def __init__(self, args, feat_in, feat_out, spread=1, bias=False):
        super(NewPoolTb, self).__init__()
        self.manifold = PoincareBall()
        self.c = args.c
        self.L_flag = torch.zeros(1, 1).cuda()
        self.bias = bias
        self.ratio = 0.8
        self.updater = nn.Linear(feat_in, feat_out, bias=self.bias)
        self.p_leader = nn.Linear(feat_out, 2, bias=self.bias)
        self.layer_weight = nn.Linear(2 * feat_out, 1, bias=self.bias)
        self.aggregator = PROPAGATION_OUT()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, T=1):
        x_tan = self.manifold.logmap0(x, c=self.c)
        edge_attr = None
        x_information_score = self.calc_information_score(x_tan, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        # random_prob = F.softmax(score, dim=-1)

        # self.prob_i = random_prob.unsqueeze(1)


        # PRELIMINARY CALCULATIONS
        self.L_flag = torch.zeros(1, 1).cuda()

        self.L_flag = self.L_flag * 0
        updated_x = F.leaky_relu(self.updater(x_tan))
        sum_Neigh_x = self.aggregator(updated_x, edge_index)

        #  SELECTION  <==============================================================
        # random_prob = F.relu(self.p_leader(sum_Neigh_x))
        # random_prob = F.softmax(random_prob, dim=-1)
        #
        # self.prob_i = random_prob[:, 1].unsqueeze(1)
        values, indices = score.topk(int(len(score) * self.ratio), dim=0, largest=True, sorted=True)
        T = torch.min(values)
        hot_prob = torch.where(score > T, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        SEL_v = hot_prob.view(-1, 1)
        self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        sum_SEL_x = self.aggregator(SEL_v * updated_x, edge_index)
        concat_sums = torch.cat([sum_SEL_x, sum_Neigh_x], dim=-1)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        A_x = F.relu(self.aggregator(weight_SEL_v * SEL_v * updated_x, edge_index))

        out = updated_x + A_x
        out = self.manifold.proj(self.manifold.expmap0(out, self.c), self.c)
        return out



class NewPool2(torch.nn.Module):
    def __init__(self, args, feat_in, feat_out, spread=1, bias=False):
        super(NewPool2, self).__init__()
        self.manifold = PoincareBall()
        self.c = args.c
        self.L_flag = torch.zeros(1, 1).cuda()
        self.bias = bias
        self.ratio = 0.8
        self.updater = nn.Linear(feat_in, feat_out, bias=self.bias)
        self.p_leader = nn.Linear(feat_out, 2, bias=self.bias)
        self.layer_weight = nn.Linear(2 * feat_out, 1, bias=self.bias)
        self.aggregator = PROPAGATION_OUT()
        self.calc_information_score = NodeInformationScore()

    def forward(self, x, edge_index, T=0.48):
        x = self.manifold.logmap0(x, c=self.c)
        # PRELIMINARY CALCULATIONS
        self.L_flag = torch.zeros(1, 1).cuda()

        depth = 0
        self.L_flag = self.L_flag * 0
        updated_x = F.leaky_relu(self.updater(x))
        sum_Neigh_x = self.aggregator(updated_x, edge_index)

        #  SELECTION  <==============================================================
        random_prob = F.relu(self.p_leader(sum_Neigh_x))
        random_prob = F.softmax(random_prob, dim=-1)

        self.prob_i = random_prob[:, 1].unsqueeze(1)
        hot_prob = torch.where(random_prob[:, 1] > T, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        SEL_v = hot_prob.view(-1, 1)
        self.L_flag = self.L_flag.float().view(-1, 1) + SEL_v

        #  SUMMATION + CONCAT <========================================================
        sum_SEL_x = self.aggregator(SEL_v * updated_x, edge_index)
        concat_sums = torch.cat([sum_SEL_x, sum_Neigh_x], dim=-1)

        #  WEIGHT             <========================================================
        weight_SEL_v = torch.sigmoid(self.layer_weight(concat_sums))
        A_x = F.relu(self.aggregator(weight_SEL_v * SEL_v * updated_x, edge_index))

        out = updated_x + A_x
        out = self.manifold.proj(self.manifold.expmap0(out, self.c), self.c)
        return out


class PROPAGATION_OUT(MessagePassing):
    def __init__(self):
        super(PROPAGATION_OUT, self).__init__()

    def forward(self, x, edge_index): return self.propagate(edge_index, x=x)

    def message(self, x_j): return x_j

    def update(self, aggr_out): return aggr_out
