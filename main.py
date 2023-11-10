import copy
from datetime import datetime

# import geoopt

from utils import setup
from argparse import ArgumentParser
import os
import torch

from training import *


def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    frame2 = run_selftrain(clients, server, local_epoch)
    outfile = os.path.join(args.outpath,
                           f'{args.dataset}_loss_localupdate_HGCN_client{args.num_clients}_local{args.local_epoch}_{args.test_time}_{args.name}.csv')
    frame2.to_csv(outfile)
    print(f"Wrote to file: {outfile}")
    # for k, v in allAccs.items():
    #     df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    # print(df)
    # for k, v in allAccs.items():
    #     df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    # print(df)
    # if config.repeat is None:
    #     outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    # else:
    #     outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    # df.to_csv(outfile)
    # print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server, model, name):
    print("Running FedAvg With {model}...")
    filename = name
    poolratio = args.poolratio
    frame1, frame2 = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    # print(frame)
    outfile = os.path.join(args.outpath + '/' + args.dataset,
                           f'{args.dataset}_fedavg_{model}_client{args.num_clients}_{poolratio}_{filename}_300_{args.name}.csv')
    frame1.to_csv(outfile)
    # outfile = os.path.join(args.outpath+'/'+args.dataset, f'{args.dataset}_loss_fedavg_HGCN_client{args.num_clients}_local{args.local_epoch}_{args.test_time}_{args.name}.csv')
    # frame2.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_time", type=int, default="1")
    parser.add_argument("--name", help='',
                        type=str, default='{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now()))
    parser.add_argument("--data_type", help='graph_base',
                        type=str, default='not_graph')
    parser.add_argument("--datapath", help='data path',
                        type=str, default="./data")
    parser.add_argument("--outpath", default='./adddirichlet', type=str)
    parser.add_argument("--dataset", help='data set:PROTEINS/Fingerprint/NCI1/IMDB-BINARY',
                        type=str, default='IMDB-BINARY')
    parser.add_argument('--poolratio', type=int, default=0.8,
                        help='0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8;')
    parser.add_argument("--model", help='GCN/GIN/HGCN/proposed/HGPSL/node select',
                        type=str, default='node select')

    parser.add_argument('--num_rounds', type=int, default=1000,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)

    parser.add_argument("--hid_dim", help='dimension of hidden layer neurons',
                        type=int, default=64)
    parser.add_argument("--num_layers", help='Model layer',
                        type=int, default=3)
    parser.add_argument("--c", help="Curvature of hyperbolic space",
                        type=float, default=1)

    parser.add_argument("--dropout", help='drop out',
                        type=float, default=0.1)
    parser.add_argument("--batch_size", help='batch size',
                        type=int, default=64)
    parser.add_argument("--wd", help='weight decay',
                        type=float, default=5e-4)
    parser.add_argument("--lr", help='learning rate',
                        type=int, default=0.01)
    parser.add_argument("--SEED", help='seed', type=int, default=5678)
    parser.add_argument("--manifold", type=str, default="PoincareBall")
    parser.add_argument("--use_bias", type=bool, default=False)
    parser.add_argument("--adam_betas", help='', type=str, default="0.9,0.999")
    parser.add_argument("--act", help='', default='relu')
    parser.add_argument("--task", help='', default='relu')
    parser.add_argument("--bias", help='', default=1)
    parser.add_argument("--use-att", help='', default=0)
    parser.add_argument("--pos_weight", help='',
                        type=int, default=0)
    parser.add_argument("--sgd", help='',
                        action='store_true')
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--dirichlet',
                        help='0: We dont use this method, 0.1: The noniid index is 0.1, 1: The noniid index is 1, 10: The noniid index is 10',
                        type=float, default=0.1)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_dataSplit = 123
    for i in range(10):
        time = i + 1
        # time = 9
        # 准备一个数据集并切分
        splitedData, df_stats = setup.prepareData_oneDS(
            datapath=args.datapath,
            data=args.dataset,
            num_client=args.num_clients,
            batchSize=args.batch_size,
            convert_x=args.convert_x,
            seed=seed_dataSplit,
            overlap=args.overlap,
            dirichlet=args.dirichlet
        )

        print("Done")

        id_process = os.getpid()

        # init_clients, init_server = setup.setup_devices(splitedData, args, args.model)
        # training
        # from inspect import signature
        # print("running " + str(time + 1))
        print("Model = GCN")
        init_clients, init_server = setup.setup_devices(splitedData, args, "GCN")
        process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="GCN", name=time)

        # print("Model = GIN")
        # init_clients, init_server = setup.setup_devices(splitedData, args, "GIN")
        # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="GIN", name=time)
        #
        # print("Model = HGCN")
        # init_clients, init_server = setup.setup_devices(splitedData, args, "HGCN")
        # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="HGCN", name=time)
        # #
        # # # print("Model = HGPSL")
        # # # init_clients, init_server = setup.setup_devices(splitedData, args, "HGPSL")
        # # # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="HGPSL", name=time)
        # #
        # print("Model = node select_T1")
        # init_clients, init_server = setup.setup_devices(splitedData, args, "node select")
        # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="node select_T1",
        #                name=time)
        #
        # print("Model = node select_Tb")
        # init_clients, init_server = setup.setup_devices(splitedData, args, "node select_Tb")
        # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="node select_Tb",
        #                name=time)
        #
        # print("Model = Method1")
        # init_clients, init_server = setup.setup_devices(splitedData, args, "Method1")
        # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), model="Method1",
        #                name=time)
