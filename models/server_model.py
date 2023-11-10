import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, GCNConv, GINConv

from hgcn.layers.hyp_layers import GCN
from hgcn.layers.hyplayers import HgpslPool
from hgcn.layers.layers import Linear
from hgcn.layers import hyp_layers, hyplayers
from hgcn.manifolds.poincare import PoincareBall

from layers import HGPSLPool


class server_hgcn(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hgcn, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        # self.pool1 = hyplayers.HGPSLPool(64, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        # self.pool2 = hyplayers.HGPSLPool(64, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, nclass)

class serverGIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(serverGIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()  # 定义容器
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))  # 一个序列，包含一层全连接、一层RELU、一层全连接
        self.graph_convs.append(GINConv(self.nn1))  # 加入一个图卷积层进入容器
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))  # 加入一个图卷积层进入容器

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())  # 一个序列，其中包含一层全连接、一层RELU
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))  # 一个序列，其中包含一层全连接层

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)  # 运行pre，其中为一个序列，包含一个全连接层
        # print(x)
        for i in range(len(self.graph_convs)):  # 迭代nlayer次
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

class server_HGPSLPool(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass):
        super(server_HGPSLPool, self).__init__()
        self.args = args
        self.num_features = nfeat
        self.nhid = nhid
        self.num_classes = nclass
        # self.dropout_ratio = args.dropout_ratio
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

class server_hgcn2(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hgcn2, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        # self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.pool1 = hyplayers.HypbolicPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = hyplayers.HypbolicPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, nclass)

class server_GCN(nn.Module):
    def __init__(self, args, num_features, nhid, num_classes):
        super(server_GCN, self).__init__()
        self.args = args
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.dropout_ratio = args.dropout

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)

class server_hyp_GCN3(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hyp_GCN3, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.pool1 = hyplayers.NewPool(args, self.nhid, self.nhid)
        self.pool2 = hyplayers.NewPool(args, self.nhid, self.nhid)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid, nclass)


class server_hyp_GCN4(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hyp_GCN4, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.pool1 = hyplayers.NewPool2(args, self.nhid, self.nhid)
        self.pool2 = hyplayers.NewPool2(args, self.nhid, self.nhid)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        # self.lin3 = torch.nn.Linear(self.nhid, nclass)


class server_hyp_GCN3Tb(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hyp_GCN3Tb, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0


        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.pool1 = hyplayers.NewPoolTb(args, self.nhid, self.nhid)
        self.pool2 = hyplayers.NewPoolTb(args, self.nhid, self.nhid)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid, nclass)