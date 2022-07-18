import matplotlib.pyplot as plt
import numpy as np


def load_log(filename):
    f = open(filename)
    all_loss = []
    all_acc = []
    for line in f.readlines():
        if "Epoch" in line:
            continue
        info = line.strip().split(" ")
        loss = float(info[4])
        acc = float(info[7])
        all_loss.append(loss)
        all_acc.append(acc)
    return all_loss[:1000], all_acc[:1000]


def plot_loss(loss_1, loss_2):
    plt.figure(0)
    plt.plot(loss_1, 'r')
    plt.plot(loss_2, 'g')
    plt.xticks(np.arange(0, len(loss_1)+1, 100))
    plt.rcParams['figure.figsize'] = (28, 6)
    plt.xlabel("Num of Epochs")
    # plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend(['VGG+LSTM loss', 'Resnet+GRU loss'])
    plt.show()


def plot_log(file_1, file_2):
    loss_1, _ = load_log(file_1)
    loss_2, _ = load_log(file_2)
    plot_loss(loss_1, loss_2)


if __name__ == '__main__':
    plot_log("log-lstm.txt", "log-gru.txt")
