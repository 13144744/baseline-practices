import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.distributed import reduce


def run_selftrain(clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)
    frame1, frame2 = pd.DataFrame(), pd.DataFrame()
    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        frame2.loc[client.name, 'test_acc'] = acc
        # allAccs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        # print("  > {} done.".format(client.name))

    return frame2


def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):
    client_number = 0
    for client in clients:
        client.download_from_server(server)
        client_number = client_number + 1
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)  # server做聚合

        for client in selected_clients:
            client.download_from_server(server)

        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        # one_loss, one_acc = 0.0, 0.0
        for client in clients:
            i = i + 1
            loss, acc = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc
            avg_loss = avg_loss + loss
            avg_acc = avg_acc + acc
            # if i == 1:
            #     one_loss, one_acc = loss, acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc

        print('Iteration: {:04d}'.format(c_round),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc))

        # print('Epoch: {:04d}'.format(c_round + 1), 'loss_train: {:.6f}'.format(loss_train),
        #       'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
        #       'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2
