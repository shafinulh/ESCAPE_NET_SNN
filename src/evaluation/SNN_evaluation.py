from __future__ import print_function

import os
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from models import escape_net_spiking as SNN


def plot_input_image(data_loader, save_path):
    """
    Plot a sample raw image (spatio-temporal signature) and plot a corresponding
    heatmap of the poisson spiking rates for the input image to be fed into the SNN.
    """
    poisson_spikes = SNN.PoissonGenerator()

    for i, (data, target) in enumerate(data_loader):
        # print(f"Feature batch shape: {data.size()}")
        # print(f"Labels batch shape: {target.size()}")
        img = data[i].squeeze()
        spike_inputs = poisson_spikes(data)
        spike_img = spike_inputs[i].squeeze()
        label = target[i]

        plt.imshow(img, cmap="viridis")
        if save_path != "":
            plt.savefig(os.path.join(save_path, f"raw_input_{i}.png"))
        plt.clf()
        plt.imshow(spike_img, cmap="viridis")
        if save_path != "":
            plt.savefig(os.path.join(save_path, f"raw_spike_input_{i}.png"))
        plt.clf()
        # print(f"Label: {label}")
        # print(spike_img.shape)


def plot_layerwise_spikerates(layer_spikes, conv_layers, save_path):
    """
    For a given sample, plot the spikerate frequency of neuron of each layer

    Plot the data stored in each key of model.module.spikes
    """
    # layer_spikes = model.module.spikes
    # conv_layers = model.module.conv_layers
    for layer, features in layer_spikes.items():
        if layer in conv_layers:
            fig, axes = plt.subplots(ncols=16, nrows=4, figsize=(50, 6))
            columns = 16
            for i, ax in enumerate(axes.flat):
                spikes = features[0][i].cpu()
                im = ax.imshow(spikes)
            plt.colorbar(im, ax=axes.ravel().tolist())
            plt.suptitle(
                f"{layer}Conv2D_{spikes.shape[0]}x{spikes.shape[1]}x{features[0].shape[0]}"
            )
            if save_path != "":
                plt.savefig(os.path.join(save_path, f"layer_spikes_{layer}.png"))
            plt.clf()
        else:
            plt.rcParams["figure.figsize"] = (2, 6)
            if layer == 8:
                spikes = features[0].squeeze()
                spikes = spikes.reshape(16, 16).cpu()
            elif layer == 10:
                spikes = features[0].squeeze()
                spikes = spikes.reshape(3, 1).cpu()
            # print(spikes.shape)
            plt.imshow(spikes, cmap="viridis")
            plt.colorbar()
            plt.title(f"{layer}Dense_{spikes.numel()}")
            if save_path != "":
                plt.savefig(os.path.join(save_path, f"layer_spikes_{layer}.png"))
            plt.clf()


def error_vs_time():
    """
    For converted only SNN, determine the number of timesteps required to converge to ANN performance
    """
    pass


def plot_spiketrains(spiketrains, layer_spikerates, save_path):
    """
    Plot of the neurons that are firing at each timestep of a simulation
    """
    # spiketrains = model.module.spiketrains

    for layer, spiketrain in spiketrains.items():
        fig, axs = plt.subplots(2, 1, figsize=[10, 7])
        spiketrain = spiketrain.cpu().numpy()
        for timestep in range(len(spiketrain)):
            spiking_neurons = [i for i, x in enumerate(spiketrain[timestep]) if x == 1]
            axs[0].scatter(
                [timestep] * len(spiking_neurons),
                spiking_neurons,
                marker="o",
                c="b",
                s=0.2,
            )

        axs[0].set_ylim(0, spiketrain.shape[1])
        axs[0].set_xlim(0, spiketrain.shape[0])
        axs[0].set_ylabel("Neuron Index")
        # axs[0].set_xlabel('Time')

        axs[0].set_title(f"Raster Plot of Spiketrains")

        axs[1].bar(range(spiketrain.shape[0]), np.sum(spiketrain, 1))

        # axs[1].set_ylim(0, 5)
        axs[1].set_xlim(0, spiketrain.shape[0])
        axs[1].set_ylabel("Number of Spikes")
        axs[1].set_xlabel("Time")

        axs[1].set_title(f"Spike Count at Each Timestep")

        fig.suptitle(
            f"Spike Trains for Layer {layer} consisting of {spiketrain.shape[1]} neurons.\nOverall Spiking Rate of {layer_spikerates[layer]: .3f}"
        )
        if save_path != "":
            plt.savefig(os.path.join(save_path, f"spiketrains_{layer}.png"))
        plt.clf()


def plot_neuron_spike_distribution(layer_spikes, save_path):  # takes very long on CPU
    """
    Histogram of the number of neurons spiking at a certain frequency
    """
    # layer_spikes = model.module.spikes

    for layer, features in layer_spikes.items():
        # plt.rcParams["figure.figsize"] = (3,3)
        spikes = features.flatten()
        cnt = Counter(spikes.tolist())
        # print(cnt)
        plt.hist(spikes.cpu())
        plt.title(f"Spike Histogram for Layer {layer}")
        plt.xlabel(f"Number of Spikes")
        plt.ylabel(f"Number of Neurons")
        if save_path != "":
            plt.savefig(os.path.join(save_path, f"spike_histogram_{layer}.png"))
        plt.clf()


def get_batch_avg_spikerates(
    batch_avg_spikerates, conv_layers, fc_layers, layer_spikes
):
    # conv_layers = model.module.conv_layers
    # fc_layers = model.module.fc_layers
    # spike = model.module.spikes
    res = 0
    for layer in conv_layers:  # convolutional layers
        res = 0
        torch.nn.functional.relu(layer_spikes[layer], inplace=True)
        for i, _ in enumerate(layer_spikes[layer]):
            res += torch.sum(layer_spikes[layer][i])
        res /= len(layer_spikes[layer])
        # print(res)
        batch_avg_spikerates.setdefault(layer, []).append(res.item())

    for x in fc_layers:  # linear layers
        res = 0
        torch.nn.functional.relu(layer_spikes[x], inplace=True)
        for i, _ in enumerate(layer_spikes[x]):
            res += torch.sum(layer_spikes[x][i])
        res /= len(layer_spikes[x])
        # print(res)
        batch_avg_spikerates.setdefault(x, []).append(res.item())

    return batch_avg_spikerates


def get_avg_spikerates(batch_avg_spikerates, num_samples):
    layer_dict = {0: "conv1", 3: "conv2", 6: "conv3", 8: "linear1", 10: "linear2"}
    avg_spikerates = {}
    neurons = [64 * 56 * 100, 64 * 28 * 50, 64 * 14 * 25, 256, 3]
    print("\n data averaged over {0} samples:".format(num_samples))
    for i, layer in enumerate(batch_avg_spikerates.keys()):
        avg_spikes = sum(batch_avg_spikerates[layer]) / len(batch_avg_spikerates[layer])
        avg_spikerate = avg_spikes / neurons[i]
        avg_spikerates.setdefault(layer, avg_spikerate)
        print(
            f"\n\t{layer_dict[layer]}: {avg_spikes:.0f} spikes, {neurons[i]} neurons, {avg_spikerate:.3f} avg spiking rate"
        )
    print(f"\n")
    return avg_spikerates


def model_energy(model):
    """
    Giving the avg spikerate of each layer, calculate the theoretical energy savings using
    empirical data found in literature
    """
    e_mac = 3.1
    e_ac = 0.1
    e_cnn, e_snn = 0, 0
    e_cnn_layer, e_snn_layer = 0, 0
    for layer in model:
        if layer.startswith("conv"):
            (k, h, w, c_i, c_o, avg_spike) = model[layer]
            e_cnn_layer = c_i * c_o * k * k * h * w * e_mac
            e_cnn += e_cnn_layer
            e_snn_layer = (avg_spike) * (e_ac / e_mac) * e_cnn_layer
            e_snn += e_snn_layer
            print(layer, "neurons", c_o * h * w)
            print(layer, "energy savings:", e_cnn_layer / e_snn_layer)

    print("total energy savings: ", e_cnn / e_snn)
    return
