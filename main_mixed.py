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
    allAccs = run_selftrain(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v


def process_fedavg_HGCN(clients, server):
    print("Running FedAvg ...")
    frame1, frame2 = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    # print(frame)
    outfile = os.path.join(args.outpath + '/' + args.dataset,
                           f'{args.dataset}_fedavg_HGCN_client{args.num_clients}_{args.test_time}_{args.name}.csv')
    frame1.to_csv(outfile)
    outfile = os.path.join(args.outpath + '/' + args.dataset,
                           f'{args.dataset}_loss_fedavg_HGCN_client{args.num_clients}_local{args.local_epoch}_{args.test_time}_{args.name}.csv')
    frame2.to_csv(outfile)
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
    parser.add_argument("--outpath", default='./log', type=str)
    parser.add_argument("--dataset", help='data set',
                        type=str, default='biochem')

    parser.add_argument('--num_rounds', type=int, default=500,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=30)

    parser.add_argument("--model", help='model',
                        type=str, default='Nostradamus')
    parser.add_argument("--seq_len", help='the length of the gradient norm sequence',
                        type=int, default=10)
    parser.add_argument("--emb_dim", help='dimension of embedded layer neurons',
                        type=int, default=10)
    parser.add_argument("--hid_dim", help='dimension of hidden layer neurons',
                        type=int, default=64)
    parser.add_argument("--mlp_dim", help='full connection layer input dimension',
                        type=int, default=16)
    parser.add_argument("--num_layers", help='Model layer',
                        type=int, default=2)
    parser.add_argument("--c", help="Curvature of hyperbolic space",
                        type=float, default=1)

    parser.add_argument("--dropout", help='drop out',
                        type=float, default=0.1)
    parser.add_argument("--alpha", help='标志和时间损失的折中（这个参数应该是不需要的）',
                        type=float, default=0.5)  # tradeoff between time and mark loss
    parser.add_argument("--batch_size", help='批量大小',
                        type=int, default=64)
    parser.add_argument("--event_class", help='事件类型数量（这个参数同样不需要）',
                        type=int, default=22)
    parser.add_argument("--verbose_step", help='详细步骤？',
                        type=int, default=322)
    parser.add_argument("--wd", help='weight decay',
                        type=float, default=5e-4)
    parser.add_argument("--lr", help='learning rate',
                        type=int, default=0.001)
    parser.add_argument("--epochs", help='epochs',
                        type=int, default=50)
    parser.add_argument("--SEED", help='seed',
                        type=int, default=5678)
    parser.add_argument("--manifold", help='嵌入图形',
                        type=str, default="PoincareBall")
    parser.add_argument("--use_bias", help='是否使用偏移量',
                        type=bool, default="TRUE")
    parser.add_argument("--adam_betas", help='',
                        type=str, default="0.9,0.999")
    parser.add_argument("--act", help='',
                        default='relu')
    parser.add_argument("--task", help='',
                        default='relu')
    parser.add_argument("--bias", help='',
                        default=1)
    parser.add_argument("--use-att", help='',
                        default=0)
    parser.add_argument("--pos_weight", help='',
                        type=int, default=0)
    parser.add_argument("--sgd", help='',
                        action='store_true')
    parser.add_argument("--cell_type", help='',
                        choices=("hyp_gru", "eucl_rnn", "eucl_gru"), default="hyp_gru")
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_dataSplit = 123

    weight = np.ones(args.event_class)

    # 准备一个数据集并切分
    splitedData, df_stats = setup.prepareData_mixDS(
        args.datapath,
        args.dataset,
        args.batch_size,
        convert_x=args.convert_x,
        seed=seed_dataSplit
    )

    print("Done")

    id_process = os.getpid()

    init_clients, init_server = setup.setup_devices(splitedData, args)

    # process_selftrain(init_clients, init_server, local_epoch=500)
    # process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=50)
    process_fedavg_HGCN(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))

