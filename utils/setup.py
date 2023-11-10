import random
from random import choices

from models.client_model import *
from models.client import *
from models.server import *
from models.server_model import *
from utils.util import *
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree


def reset_batch_adj(data):
    print(data['train'].dataset)
    print(data['train'].dataset[0])
    adj_train = data['train'].dataset
    pass
    # return adj


# 分块，需要图数据、client数量、客户端数据是否有重叠、随机种子
def _randChunk(graphs, num_client, overlap, seed=None, dirichlet=0):
    random.seed(seed)
    np.random.seed(seed)
    if dirichlet == 0:
        totalNum = len(graphs)  # 计算graphs总数
        minSize = min(50, int(totalNum / num_client))  # 平均分配并取最小值，如果最小值大于50，那么以50作为一个client中数据量
        graphs_chunks = []  # 数据分块
        if not overlap:
            for i in range(num_client):
                graphs_chunks.append(graphs[i * minSize:(i + 1) * minSize])
            for g in graphs[num_client * minSize:]:
                idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
                graphs_chunks[idx_chunk].append(g)
        else:
            sizes = np.random.randint(low=50, high=150, size=num_client)
            for s in sizes:
                graphs_chunks.append(choices(graphs, k=s))
        return graphs_chunks
    else:
        minSize = int(min(50, int(len(graphs) / num_client)) / 2)  # 平均分配并取最小值，如果最小值大于50，那么以50作为一个client中数据量
        begin_graph = graphs[:num_client * minSize]
        mid = graphs[num_client * minSize:]
        mid_graph = mid[:len(mid) - num_client * minSize]
        end_graph = mid[len(mid) - num_client * minSize:]
        graphs_chunks = []  # 数据分块
        if not overlap:
            for i in range(num_client):
                graphs_chunks.append(begin_graph[i * minSize:(i + 1) * minSize])

            idx = np.random.randint(low=0, high=num_client, size=1)[0]
            for x in range(idx):
                random.shuffle(mid_graph)
            rd = np.random.dirichlet(np.array((num_client * [dirichlet])))
            totalNum = len(mid_graph) * rd
            used_chunks = 0
            for i in range(num_client):
                num_chunks = int(totalNum[i])
                begin = used_chunks
                end = used_chunks + num_chunks
                for g in mid_graph[begin:end]:
                    graphs_chunks[i].append(g)
                used_chunks = end
            for g in mid_graph[used_chunks:]:
                idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
                graphs_chunks[idx_chunk].append(g)

            for i in range(num_client):
                for g in end_graph[i * minSize:(i + 1) * minSize]:
                    graphs_chunks[i].append(g)
        else:
            sizes = np.random.randint(low=50, high=150, size=num_client)
            for s in sizes:
                graphs_chunks.append(choices(graphs, k=s))
        return graphs_chunks


def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False, dirichlet=0):
    use_node_attr, use_edge_attr = False, False
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                              pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                              pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                              pre_transform=OneHotDegree(88, cat=False))
    elif data == "Letter-high":
        use_node_attr, use_edge_attr = True, True
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
    elif data == "Fingerprint":
        use_node_attr, use_edge_attr = True, False
        tudataset = TUDataset(f"{datapath}/TUDataset", "Fingerprint", use_node_attr=use_node_attr,
                              use_edge_attr=use_edge_attr)
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    # graphs = [x for x in tudataset]
    graphs = []

    for x in tudataset:
        graphs.append(x)
    print("  **", data, len(graphs))
    # graphs = mess_up_dataset(graphs, 0.2)
    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed, dirichlet=dirichlet)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    num_graph_labels = get_numGraphLabels(graphs)
    print(num_graph_labels)
    add_noise_clients = num_client / 4
    for idx, chunks in enumerate(graphs_chunks):
        if idx <= add_noise_clients:
            chunks = mess_up_dataset(chunks, 0.2, use_edge_attr, data)

        ds = f'{idx}-{data}'
        ds_tvt = chunks
        # graphs_train, graphs_valtest = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        # graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        graphs_train, graphs_val, graphs_test = split(ds_tvt)


        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)


        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df


def prepareData_mixDS(datapath, dataset='molecules', batchSize=32, convert_x=False, seed=None):
    assert dataset in ['social', 'molecules', 'molecules_tiny', 'small', 'mix', "mix_tiny", "biochem", "biochem_tiny"]

    if dataset == 'social':
        datasets = ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]
    if dataset == 'molecules' or dataset == 'molecules_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if dataset == 'small':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR",  # small molecules
                    "DD", "PROTEINS"]  # bioinformatics
    if dataset == 'mix' or dataset == 'mix_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",  # bioinformatics
                    "IMDB-BINARY", "IMDB-MULTI"]  # social networks
    if dataset == 'biochem' or dataset == 'biochem_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)
            if convert_x:
                maxdegree = get_maxDegree(tudataset)
                tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        if dataset.endswith('tiny'):
            graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
            graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    return splitedData, df


def prepareData_fingerprint(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False, dirichlet=0):
    use_node_attr, use_edge_attr = False, False
    if data == "Fingerprint":
        use_node_attr = True
        tudataset = TUDataset(f"{datapath}/TUDataset", "Fingerprint", use_node_attr=use_node_attr,
                              use_edge_attr=use_edge_attr)
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    # graphs = [x for x in tudataset]
    graphs = []

    for x in tudataset:
        graphs.append(x)
    print("  **", data, len(graphs))
    # graphs = mess_up_dataset(graphs, 0.2)
    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed, dirichlet=dirichlet)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    num_graph_labels = get_numGraphLabels(graphs)
    for idx, chunks in enumerate(graphs_chunks):
        # if idx <= add_noise_clients:
        #     chunks = mess_up_dataset(chunks, 0.2, use_edge_attr, data)
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        # graphs_train, graphs_valtest = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        # graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        graphs_train, graphs_val, graphs_test, _ = fingerprint_split(ds_tvt, seed=seed)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df


def setup_devices(splitedData, args, use_model):
    idx_clients = {}
    clients = []
    # use_model = "HGCN"  # "HGCN" "GIN" "HGPSLPool" "proposed"
    if use_model == "HGCN":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = hyp_GCN(args.manifold,
                             num_node_features,
                             args.hid_dim,
                             num_graph_labels,
                             args.num_layers,
                             args.dropout,
                             args
                             )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args))
        smodel = server_hgcn(args.manifold,
                             num_node_features,
                             args.hid_dim,
                             num_graph_labels,
                             args.num_layers,
                             args.dropout,
                             args)
        server = Server_Net(smodel, args.device, args)
    elif use_model == "GIN":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = GIN(nfeat=num_node_features,
                         nhid=args.hid_dim,
                         nclass=num_graph_labels,
                         nlayer=args.num_layers,
                         dropout=args.dropout)
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(
                Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
            )
        smodel = serverGIN(nfeat=num_node_features,
                           nhid=args.hid_dim,
                           nclass=num_graph_labels,
                           nlayer=args.num_layers,
                           dropout=args.dropout)
        server = Server_Net(smodel, args.device, args)
    elif use_model == "HGPSL":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = client_HGPSLPool(args, num_node_features, args.hid_dim, 15, )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(
                Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
            )
        smodel = server_HGPSLPool(args, num_node_features, args.hid_dim, 15, )
        server = Server_Net(smodel, args.device, args)
    elif use_model == "GCN":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = client_GCN(args, num_node_features, args.hid_dim, num_graph_labels, )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(
                Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
            )
        smodel = server_GCN(args, num_node_features, args.hid_dim, num_graph_labels, )
        server = Server_Net(smodel, args.device, args)
    # elif use_model == "proposed":
    #     for idx, ds in enumerate(splitedData.keys()):
    #         print(ds + " preparing")
    #         idx_clients[idx] = ds
    #         dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
    #         print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
    #         cmodel = hyp_GCN2(args.manifold,
    #                           num_node_features,
    #                           args.hid_dim,
    #                           num_graph_labels,
    #                           args.num_layers,
    #                           args.dropout,
    #                           args
    #                           )
    #         optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
    #         clients.append(
    #             Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
    #         )
    #     smodel = server_hgcn2(args.manifold,
    #                           num_node_features,
    #                           args.hid_dim,
    #                           num_graph_labels,
    #                           args.num_layers,
    #                           args.dropout,
    #                           args
    #                           )
    #     server = Server_Net(smodel, args.device, args)
    elif use_model == "node select":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = hyp_GCN3(args.manifold,
                              num_node_features,
                              args.hid_dim,
                              num_graph_labels,
                              args.num_layers,
                              args.dropout,
                              args
                              )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(
                Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
            )
        smodel = server_hyp_GCN3(args.manifold,
                                 num_node_features,
                                 args.hid_dim,
                                 15,
                                 args.num_layers,
                                 args.dropout,
                                 args
                                 )
        server = Server_Net(smodel, args.device, args)
    elif use_model == "Method1":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = hyp_GCN4(args.manifold,
                              num_node_features,
                              args.hid_dim,
                              num_graph_labels,
                              args.num_layers,
                              args.dropout,
                              args
                              )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(
                Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
            )
        smodel = server_hyp_GCN4(args.manifold,
                                 num_node_features,
                                 args.hid_dim,
                                 15,
                                 args.num_layers,
                                 args.dropout,
                                 args
                                 )
        server = Server_Net(smodel, args.device, args)
    elif use_model == "node select_Tb":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = hyp_GCN3Tb(args.manifold,
                              num_node_features,
                              args.hid_dim,
                              num_graph_labels,
                              args.num_layers,
                              args.dropout,
                              args
                              )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(
                Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args)
            )
        smodel = server_hyp_GCN3Tb(args.manifold,
                                 num_node_features,
                                 args.hid_dim,
                                 num_graph_labels,
                                 args.num_layers,
                                 args.dropout,
                                 args
                                 )
        server = Server_Net(smodel, args.device, args)
    return clients, server
