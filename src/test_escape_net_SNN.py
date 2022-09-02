# ---------------------------------------------------
# Imports
# ---------------------------------------------------
from __future__ import print_function

import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import make_dataset
from evaluation.SNN_evaluation import (
    get_avg_spikerates,
    get_batch_avg_spikerates,
    model_energy,
    plot_input_image,
    plot_layerwise_spikerates,
    plot_neuron_spike_distribution,
    plot_spiketrains,
)
from models import escape_net_spiking as SNN
from utils import AverageMeter, CustomDataset, cli


def make_data_loader(data, labels, batch_size, transforms):
    dataset = CustomDataset.CustomDataset(data, labels, transforms)

    if torch.cuda.is_available() and args.gpu:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def evaluate_sample(samples_to_test=1):
    # obtain the sample set given the num samples to evaluate
    indices = np.random.choice(
        RAT_data.test_set.shape[0], samples_to_test, replace=False
    )
    sample_data = RAT_data.test_set[indices]
    sample_labels = RAT_data.test_labels[indices]
    sample_loader = make_data_loader(
        sample_data, sample_labels, samples_to_test, transforms.ToTensor()
    )

    # visualize the input images and input spiking rate maps
    plot_input_image(sample_loader, save_path)

    # make the prediction and store the number of spikes in each layer
    layer_spikerates = inference(sample_loader)

    # get the appropriate data from the model variables to evaluate
    conv_layers = model.module.conv_layers
    fc_layers = model.module.fc_layers
    layer_spikes = model.module.spikes
    spiketrains = model.module.spiketrains

    # create appropriate figures based on the inference of single image
    plot_layerwise_spikerates(layer_spikes, conv_layers, save_path)
    plot_spiketrains(spiketrains, layer_spikerates, save_path)
    plot_neuron_spike_distribution(layer_spikes, save_path)

    """manually create the model dictionary 
        conv_layer: (filt_size, height, width, in_features, out_features, layer_avg_spikerate)
        fc_layer: (in_features, out_features)
    """
    escape_net = {}
    escape_net["conv1"] = (8, 56, 100, 1, 64, layer_spikerates[0])
    escape_net["conv2"] = (4, 28, 50, 64, 64, layer_spikerates[3])
    escape_net["conv3"] = (2, 14, 25, 64, 64, layer_spikerates[6])
    escape_net["linear1"] = (22400, 256, layer_spikerates[8])
    escape_net["linear2"] = (256, 3, layer_spikerates[10])

    model_energy(escape_net)


def inference(data_loader):
    global history
    global batch_avg_spikerates

    batch_avg_spikerates = {}
    targets, preds = [], []
    conv_layers = model.module.conv_layers
    fc_layers = model.module.fc_layers

    losses = AverageMeter.AverageMeter("Loss")
    top1 = AverageMeter.AverageMeter("Acc@1")

    with torch.no_grad():
        model.eval()
        global max_accuracy

        for batch_idx, (data, target) in enumerate(data_loader):

            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            target = target.long()
            loss = F.cross_entropy(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(), data.size(0))
            top1.update(correct.item() / data.size(0), data.size(0))

            if test_acc_every_batch:

                f.write(
                    "\nAccuracy: {}/{}({:.4f})".format(
                        correct.item(), data.size(0), top1.avg
                    )
                )
            layer_spikes = model.module.spikes
            batch_avg_spikerates = get_batch_avg_spikerates(
                batch_avg_spikerates, conv_layers, fc_layers, layer_spikes
            )
            targets.extend(target.cpu().tolist())
            pred = pred.data.view_as(target).cpu()
            preds.extend(pred.cpu().tolist())

        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1 + [value.item()]

        f.write(
            "\n test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}".format(
                losses.avg,
                top1.avg,
                max_accuracy,
                datetime.timedelta(seconds=(datetime.now() - start_time).seconds),
            )
        )
        # Calculate the spikerate of each layer averaged over all samples
        avg_spikerates = get_avg_spikerates(
            batch_avg_spikerates, len(data_loader.dataset)
        )
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        return avg_spikerates


if __name__ == "__main__":
    args = cli.test_SNN_arg_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # Seed random number
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = args.dataset
    batch_size = args.batch_size
    architecture = args.architecture
    pretrained_snn = args.pretrained_snn
    epochs = args.epochs
    timesteps = args.timesteps
    leak = args.leak
    default_threshold = args.default_threshold
    activation = args.activation
    alpha = args.alpha
    beta = args.beta
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    amsgrad = args.amsgrad
    dropout = args.dropout
    kernel_size = args.kernel_size
    test_acc_every_batch = args.test_acc_every_batch
    train_acc_batches = args.train_acc_batches
    dataset_path = args.dataset_path
    save_dir = args.save_dir
    save_name = args.save_name

    version = 0
    save_path = os.path.join(save_dir, save_name)

    save = False if save_path == "" else False

    while True and save:
        try:
            os.mkdir(os.path.join(save_path, "_", version))
            break
        except OSError:
            version += 1

    log_file = save_name + ".log" if save else ""
    log_file = os.path.join(save_path, log_file)

    if log_file and args.log:
        f = open(log_file, "w", buffering=1)
    else:
        f = sys.stdout

    f.write("\n Run on time: {}".format(datetime.now()))

    f.write("\n\n Arguments: ")
    for arg in vars(args):
        f.write("\n\t {:20} : {}".format(arg, getattr(args, arg)))

    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if dataset == "RAT4":
        labels = 3
        RAT_data = make_dataset.Preprocessing_module("Rat4", 1, dataset_path)
        train_loader = make_data_loader(
            RAT_data.training_set,
            RAT_data.training_labels,
            batch_size,
            transforms.ToTensor(),
        )
        test_loader = make_data_loader(
            RAT_data.test_set[0:5],
            RAT_data.test_labels[0:5],
            batch_size,
            transforms.ToTensor(),
        )

    model = SNN.ESCAPE_NET_SNN_STDB(
        model_name="ESCAPE_NET",
        activation=activation,
        labels=labels,
        timesteps=timesteps,
        leak=leak,
        default_threshold=default_threshold,
        alpha=alpha,
        beta=beta,
        dropout=dropout,
        kernel_size=kernel_size,
        dataset=dataset,
    )
    model = nn.DataParallel(model)
    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    if pretrained_snn:
        state = torch.load(pretrained_snn, map_location="cpu")
        cur_dict = model.state_dict()
        f.write("\n Info: Pretrained SNN data")
        for key in state.keys():
            if key != "thresholds" and key != "state_dict" and key != "optimizer":
                f.write("\n \t {} : {}".format(key, state[key]))
        for key in state["state_dict"].keys():

            if key in cur_dict:
                if state["state_dict"][key].shape == cur_dict[key].shape:
                    cur_dict[key] = nn.Parameter(state["state_dict"][key].data)
                    f.write("\n Loaded {} from {}".format(key, pretrained_snn))
                else:
                    f.write(
                        "\n Size mismatch {}, size of loaded model {}, size of current model {}".format(
                            key,
                            state["state_dict"][key].shape,
                            model.state_dict()[key].shape,
                        )
                    )
            else:
                f.write("\n Loaded weight {} not present in current model".format(key))
        model.load_state_dict(cur_dict)

        if "thresholds" in state.keys():
            try:
                if state["leak_mem"]:
                    state["leak"] = state["leak_mem"]
            except:
                pass
            if state["timesteps"] != timesteps or state["leak"] != leak:
                f.write(
                    "\n Timesteps/Leak mismatch between loaded SNN and current simulation timesteps/leak, current timesteps/leak {}/{}, loaded timesteps/leak {}/{}".format(
                        timesteps, leak, state["timesteps"], state["leak"]
                    )
                )
            thresholds = state["thresholds"]
            model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
        else:
            f.write("\n Loaded SNN model does not have thresholds")

    f.write("\n {}".format(model))

    max_accuracy = 0
    avg_spike_dict = {}

    start_time = datetime.now()

    evaluate_sample(1)
    # avg_spikerates = inference(test_loader)
