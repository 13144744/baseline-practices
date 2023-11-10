import time

from utils.data_utils import *
import torch.nn.functional as F
from torch import nn
from utils.util import *


def RMSE_error(pred, gold):
    return np.sqrt(np.mean((pred - gold) ** 2))


class Client_Net(nn.Module):
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        super().__init__()
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.c = args.c

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        train_stats = train_client(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        # grads = {key: value.grad for key, value in self.W.items() and self.W.items()}
        # self.gradsNorm = torch.norm(flatten(grads)).item()
        # for key, value in self.W.items():
        #     if value.grad != None:
        #         grads.

        # grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        # self.convGradsNorm = torch.norm(flatten(grads_conv)).item()



    def evaluate(self):
        return eval_local(self.model, self.dataLoader['test'], self.args.device)


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])


def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


# def reset_batch_adj(databatch):
#     edge_index, x, y, batch, ptr = databatch.edge_index, databatch.x, databatch.y, databatch.batch, databatch.ptr
#     adj = process(edge_adj(edge_index, x), True)
#     return adj


def train_client(model, dataloaders, optimizer, local_epoch, device, clientNo):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_local(model, val_loader, device)
        loss_tt, acc_tt = eval_local(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)
    return {'trainingLosses': losses_train, 'trainingAccs': accs_train,
            'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}


def eval_local(model, test_loader, device):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs
    return total_loss / ngraphs, acc_sum / ngraphs
