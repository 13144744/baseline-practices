import os

import pandas as pd
from matplotlib import pyplot as plt


def showAccs(inpath):
    str = "NCI1_fedavg_HGCN_client30_2022_04_10_15_10_54_518915"
    df_self = pd.read_csv(os.path.join(inpath, str + ".csv"), header=0, index_col=0)
    print(df_self)
    plt.title("Epoch_Accs")
    x = range(len(df_self))
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.plot(x, df_self['avg_acc'], color="blue", label="avg_acc")
    plt.plot(x, df_self['one_acc'], color="lightblue", label="one_acc")
    plt.plot(x, df_self['server_accs'], color="red", label="server_accs")
    plt.savefig(str + '_accs')
    print("done")
    # print(f"Wrote to file: {outfile_jpg}")
    plt.show()


def showLosses(inpath):
    str = "accuracy_fedavg_HGCN2022_04_09_00_13_25_336116"
    df_self = pd.read_csv(os.path.join(inpath, str + ".csv"), header=0, index_col=0)
    print(df_self)
    plt.title("Epoch_Loss")
    x = range(len(df_self))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, df_self['avg_loss'], color="lightgreen")
    plt.plot(x, df_self['one_loss'], color="lightblue")
    plt.plot(x, df_self['server_losses'], color="red")
    plt.savefig(str + '_loss')
    print("done")
    # print(f"Wrote to file: {outfile_jpg}")
    plt.show()


def showAccsControlGroup(inpath):
    str = "accuracy_fedavg_GC"
    df_self = pd.read_csv(os.path.join(inpath, str + ".csv"), header=0, index_col=0)
    print(df_self)
    plt.title("Epoch_Accs")
    x = range(len(df_self))
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.plot(x, df_self['avg_acc'])
    plt.savefig(str + '_acc')
    # print(f"Wrote to file: {outfile_jpg}")
    # plt.show()


def showLossesControlGroup(inpath):
    str = "accuracy_fedavg_GC"
    df_self = pd.read_csv(os.path.join(inpath, str + ".csv"), header=0, index_col=0)
    print(df_self)
    plt.title("Epoch_Loss")
    x = range(len(df_self))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, df_self['avg_loss'])
    plt.savefig(str + '_loss')
    # print(f"Wrote to file: {outfile_jpg}")
    # plt.show()


# if __name__ == '__main__':
#     showAccs('./log')
#     showLosses('./log')
#     # showAccsControlGroup('./log')
#     # showLossesControlGroup('./log')


from torch.utils.tensorboard import SummaryWriter
import pandas as pd
writer = SummaryWriter('./tensorboard_draw')
df = pd.read_csv('./adddirichlet/Fingerprint draw/Fingerprint_fedavg_GCN_client10_1_300_2022_09_14_14_45_54_329893.csv', header=0, index_col=0)
for a in range(df.shape[0]):
    loss = df.iat[a,0]
    accuracy = df.iat[a,1]
    writer.add_scalar('loss', loss, a)
    writer.add_scalar('accuracy', accuracy, a)

#####caculating######
