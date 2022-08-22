import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_curves(model_path):
    history_path = os.path.join(model_path, 'history.npy')
    history_dict = np.load(history_path, allow_pickle='TRUE')

    model_name = model_path.split('/')[-1]

    train_loss = history_dict[train_loss]
    test_loss = history_dict[test_loss]
    plt.plot(train_loss, 'b', label = 'Train Loss')
    plt.plot(test_loss, 'r', label = 'Test Loss')
    plt.legend(loc="best")
    plt.title(f'Loss curve for {model_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.ylim(0, 1.2)
    plt.show()


    train_acc = history_dict[train_acc]
    test_acc = history_dict[test_acc]
    plt.plot(train_acc, 'b', label = 'Train Acc')
    plt.plot(test_acc, 'r', label = 'Test Acc')
    plt.legend(loc="best")
    plt.title(f'Accuracy curve for {model_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim(0, 1)
    plt.show()