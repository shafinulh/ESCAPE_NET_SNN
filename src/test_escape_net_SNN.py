#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
from collections import Counter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
# from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import sys
import os
import shutil
import argparse
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection


import make_dataset
from models import escape_net_spiking as SNN
from utils import CustomDataset, AverageMeter


def make_data_loader(data, labels, batch_size, transforms):
    dataset = CustomDataset.CustomDataset(data, labels, transforms)

    if torch.cuda.is_available() and args.gpu:
        data_loader    = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        data_loader    = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def evaluate_sample(samples_to_test=1):
    # obtain the sample set given the num samples to evaluate
    indices = np.random.choice(RAT_data.test_set.shape[0], samples_to_test, replace=False)
    sample_data = RAT_data.test_set[indices]
    sample_labels = RAT_data.test_labels[indices]
    sample_loader = make_data_loader(sample_data, sample_labels, samples_to_test, transforms.ToTensor())
    
    # visualize the input images and input spiking rate maps
    # input_image(sample_loader)
    
    # make the prediction and store the number of spikes in each layer
    spikerate = inference(sample_loader)

    # create appropriate figures based on the inference of single image
    # layerwise_spikerates()
    # spiketrains(spikerate)
    neuron_spike_distribution()
    
    '''manually create the model dictionary 
        conv_layer: (filt_size, height, width, in_features, out_features, layer_avg_spikerate)
        fc_layer: (in_features, out_features)
    '''
    # escape_net = {}
    # escape_net['conv1'] = (8, 56, 100, 1, 64, spikerate[0])
    # escape_net['conv2'] = (4, 28, 50, 64, 64, spikerate[3])
    # escape_net['conv3'] = (2, 14, 25, 64, 64, spikerate[6])
    # escape_net['linear1'] = (22400, 256, spikerate[8])
    # escape_net['linear2'] = (256, 3, spikerate[10])

    # model_energy(escape_net)

def inference(data_loader):
    global history
    global batch_avg_spikerates

    batch_avg_spikerates = {}
    targets, preds = [], []

    losses = AverageMeter.AverageMeter('Loss')
    top1   = AverageMeter.AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(data_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output  = model(data) 
            print(output.dtype, target.dtype)
            target = target.long()
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))

            if test_acc_every_batch:
                
                f.write('\nAccuracy: {}/{}({:.4f})'
                    .format(
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )

            batch_avg_spikerates = get_batch_avg_spikerates(batch_avg_spikerates)
            targets.extend(target.cpu().tolist())
            pred = pred.data.view_as(target).cpu()
            preds.extend(pred.cpu().tolist())
        
        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1+[value.item()]    

        f.write('\n test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )
        # Calculate the spikerate of each layer averaged over all samples 
        avg_spikerates = get_avg_spikerates(batch_avg_spikerates, len(data_loader.dataset))
        get_confusion_matrix(targets, preds)
        return avg_spikerates

def get_confusion_matrix(targets, preds):
    '''
    basic plot of the trained SNN's predictions. COMPARE to the corresponding ANN
    '''
    classes = {0: 'A', 1: '2', 2: '3'}
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def input_image(data_loader):
    '''
    Plot a sample raw image (spatio-temporal signature) and plot a corresponding 
    heatmap of the poisson spiking rates for the input image to be fed into the SNN.
    '''
    poisson_spikes = SNN.PoissonGenerator()

    for i, (data, target) in enumerate(data_loader):
        # print(f"Feature batch shape: {data.size()}")
        # print(f"Labels batch shape: {target.size()}")
        img = data[i].squeeze()
        spike_inputs = poisson_spikes(data)
        spike_img = spike_inputs[i].squeeze()
        label = target[i]

        plt.imshow(img, cmap="viridis")
        plt.savefig(os.path.join(save_path, f'raw_input_{i}.png'))
        plt.clf()
        plt.imshow(spike_img, cmap="viridis")
        plt.savefig(os.path.join(save_path, f'raw_spike_input_{i}.png'))
        plt.clf()
        # print(f"Label: {label}")
        # print(spike_img.shape)

def layerwise_spikerates():
    '''
    For a given sample, plot the spikerate frequency of neuron of each layer

    Plot the data stored in each key of model.module.spikes
    '''
    layer_spikes = model.module.spikes
    conv_layers = model.module.conv_layers
    for layer, features in layer_spikes.items():
        if layer in conv_layers:
            fig, axes = plt.subplots(ncols=16, nrows=4, figsize=(50,6))
            columns = 16
            for i, ax in enumerate(axes.flat):
                spikes = features[0][i].cpu()
                im = ax.imshow(spikes)
            plt.colorbar(im, ax=axes.ravel().tolist())
            plt.suptitle(f'{layer}Conv2D_{spikes.shape[0]}x{spikes.shape[1]}x{features[0].shape[0]}')
            plt.savefig(os.path.join(save_path, f'layer_spikes_{layer}.png'))
            plt.clf()
        else:
            plt.rcParams["figure.figsize"] = (2,6)
            if layer == 8:
                spikes = features[0].squeeze()
                spikes = spikes.reshape(16, 16).cpu()
            elif layer == 10:
                spikes = features[0].squeeze()
                spikes = spikes.reshape(3, 1).cpu()
            print(spikes.shape)
            plt.imshow(spikes, cmap="viridis")
            plt.colorbar()
            plt.title(f'{layer}Dense_{spikes.numel()}')
            plt.savefig(os.path.join(save_path, f'layer_spikes_{layer}.png'))
            plt.clf()

def error_vs_time():
    '''
    For converted only SNN, determine the number of timesteps required to converge to ANN performance
    '''
    pass

def spiketrains(spikerate):
    '''
    Plot of the neurons that are firing at each timestep of a simulation
    '''
    spiketrains = model.module.spiketrains

    for layer, spiketrain in spiketrains.items():
        fig, axs = plt.subplots(2, 1, figsize=[10,7])
        spiketrain = spiketrain.cpu().numpy()
        for timestep in range(len(spiketrain)):
            spiking_neurons = [i for i, x in enumerate(spiketrain[timestep]) if x == 1]
            axs[0].scatter([timestep]*len(spiking_neurons), spiking_neurons, marker='o', c = 'b', s=0.2)

        axs[0].set_ylim(0, spiketrain.shape[1])
        axs[0].set_xlim(0, spiketrain.shape[0])
        axs[0].set_ylabel('Neuron Index')
        # axs[0].set_xlabel('Time')

        axs[0].set_title(f'Raster Plot of Spiketrains')

        axs[1].bar(range(spiketrain.shape[0]), 
          np.sum(spiketrain, 1)
        )

        # axs[1].set_ylim(0, 5)
        axs[1].set_xlim(0, spiketrain.shape[0])
        axs[1].set_ylabel('Number of Spikes')
        axs[1].set_xlabel('Time')

        axs[1].set_title(f'Spike Count at Each Timestep')
        
        fig.suptitle(f'Spike Trains for Layer {layer} consisting of {spiketrain.shape[1]} neurons.\nOverall Spiking Rate of {spikerate[layer]: .3f}')
        plt.savefig(os.path.join(save_path, f'spiketrains_{layer}.png'))
        plt.clf()

def neuron_spike_distribution(): # takes very long on CPU
    '''
    Histogram of the number of neurons spiking at a certain frequency
    '''
    layer_spikes = model.module.spikes
    conv_layers = model.module.conv_layers
    
    for layer, features in layer_spikes.items():
        # plt.rcParams["figure.figsize"] = (3,3)
        spikes = features.flatten()
        cnt = Counter(spikes.tolist())
        print('here')
        # print(cnt)
        plt.hist(spikes.cpu())
        print('there')
        plt.title(f'Spike Histogram for Layer {layer}')
        plt.xlabel(f'Number of Spikes')
        plt.ylabel(f'Number of Neurons')
        plt.savefig(os.path.join(save_path, f'spike_histogram_{layer}.png'))
        plt.clf() 

def get_batch_avg_spikerates(batch_avg_spikerates):
    conv_layers = model.module.conv_layers
    fc_layers = model.module.fc_layers
    spike = model.module.spikes
    res = 0
    for layer in conv_layers: #convolutional layers
        res = 0
        torch.nn.functional.relu(spike[layer], inplace=True)
        for i, _ in enumerate(spike[layer]):
          res += torch.sum(spike[layer][i])
        res/=len(spike[layer])
        # print(res)
        batch_avg_spikerates.setdefault(layer, []).append(res.item())

    for x in fc_layers: #linear layers
        res = 0
        torch.nn.functional.relu(spike[x], inplace=True)
        for i, _ in enumerate(spike[x]):
          res += torch.sum(spike[x][i])
        res/=len(spike[x])
        # print(res)
        batch_avg_spikerates.setdefault(x, []).append(res.item())

    return batch_avg_spikerates


def get_avg_spikerates(batch_avg_spikerates, num_samples):
    layer_dict = {0: 'conv1', 3: 'conv2', 6: 'conv3', 8: 'linear1', 10: 'linear2'}
    avg_spikerates = {}
    neurons = [64*56*100, 64*28*50, 64*14*25, 256, 3]
    f.write('\n data averaged over {0} samples:'.format(num_samples))
    for i, layer in enumerate(batch_avg_spikerates.keys()):
        avg_spikes = sum(batch_avg_spikerates[layer])/len(batch_avg_spikerates[layer])
        avg_spikerate = (avg_spikes/neurons[i])
        avg_spikerates.setdefault(layer, avg_spikerate)
        f.write(f'\n\t{layer_dict[layer]}: {avg_spikes:.0f} spikes, {neurons[i]} neurons, {avg_spikerate:.3f} avg spiking rate')
    f.write(f'\n')
    return avg_spikerates

def model_energy(model):
    '''
    Giving the avg spikerate of each layer, calculate the theoretical energy savings using 
    empirical data found in literature
    '''
    e_mac = 3.1
    e_ac = 0.1
    e_cnn, e_snn = 0, 0
    e_cnn_layer, e_snn_layer = 0, 0
    for layer in model:
        if layer.startswith('conv'):
            (k, h, w, c_i, c_o, avg_spike) = model[layer]
            e_cnn_layer = c_i*c_o*k*k*h*w * e_mac
            e_cnn += e_cnn_layer
            e_snn_layer = (avg_spike) * (e_ac/e_mac) * e_cnn_layer  
            e_snn += e_snn_layer
            print(layer, 'neurons', c_o*h*w)
            print(layer, 'energy savings:', e_cnn_layer/e_snn_layer)

    print ('total energy savings: ', e_cnn/e_snn)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='RAT4',             type=str,       help='dataset name')
    parser.add_argument('--batch_size',             default=32,                 type=int,       help='minibatch size')
    parser.add_argument('--architecture',           default='ESCAPE_NET',       type=str,       help='network architecture')
    parser.add_argument('-lr',                      default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/trained_models/snn/snn_escape_net_727_28_2/snn_escape_net_727_28_1.pth',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--log',                    default=False,             type=bool,      help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=10,                 type=int,       help='number of training epochs')
    parser.add_argument('--timesteps',              default=28,                 type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='Adam',             type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')    
    parser.add_argument('--momentum',               default=0.95,               type=float,     help='momentum parameter for the SGD optimizer')    
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=8,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=50,                 type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--dataset_path',           default='D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/data/processed/Rat4Training_Fold1.mat', type=str, help = '')
    parser.add_argument('--save_dir',               default = 'D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/trained_models/snn',               type=str,       help='')
    parser.add_argument('--save_name',              default = 'snn_escape_net_727_28_2',               type=str,       help='')
    parser.add_argument('-f')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = 'ESCAPE_NET'
    pretrained_snn      = args.pretrained_snn
    # pretrained_snn      = '/content/drive/MyDrive/ISML22/Hybrid Conversion/trained_models/snn/snn_escape_net_727_28_1/snn_escape_net_727_28_1.pth'
    epochs              = args.epochs
    timesteps           = args.timesteps
    leak                = args.leak
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= False
    train_acc_batches   = args.train_acc_batches
    dataset_path        = args.dataset_path
    # dataset_path        = '/content/drive/MyDrive/ISML22/original_implementations/datasets/ryan_dataset/No_TE/Rat4Training_Fold1.mat'
    save_dir            = args.save_dir
    # save_dir            = '/content/drive/MyDrive/ISML22/Hybrid Conversion/trained_models/snn/'
    save_name           = args.save_name
    # save_name           = 'snn_escape_net_727_28_1'
    

    save_path = os.path.join(save_dir, save_name)
    try:
        os.mkdir(save_path)
    except OSError:
        pass 

    log_file = save_name + '.log'
    log_file = os.path.join(save_path, log_file)
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    f.write('\n Run on time: {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    
    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'RAT4':
        labels = 3
        RAT_data = make_dataset.Preprocessing_module('Rat4',1,dataset_path)
        train_loader = make_data_loader(RAT_data.training_set, RAT_data.training_labels, batch_size, transforms.ToTensor())
        test_loader = make_data_loader(RAT_data.test_set[0:5], RAT_data.test_labels[0:5], batch_size, transforms.ToTensor())

    model = SNN.ESCAPE_NET_SNN_STDB(model_name='ESCAPE_NET', activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, kernel_size=kernel_size, dataset=dataset)
    model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    if pretrained_snn:  
        state = torch.load(pretrained_snn, map_location='cpu')
        cur_dict = model.state_dict()     
        f.write('\n Info: Pretrained SNN data')
        for key in state.keys():
              if key != 'thresholds' and key != 'state_dict' and key != 'optimizer':
                f.write('\n \t {} : {}'.format(key, state[key]))
        for key in state['state_dict'].keys():
            
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Loaded {} from {}'.format(key, pretrained_snn))
                else:
                    f.write('\n Size mismatch {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Loaded weight {} not present in current model'.format(key))
        model.load_state_dict(cur_dict)

        if 'thresholds' in state.keys():
            try:
                if state['leak_mem']:
                    state['leak'] = state['leak_mem']
            except:
                pass
            if state['timesteps']!=timesteps or state['leak']!=leak:
                f.write('\n Timesteps/Leak mismatch between loaded SNN and current simulation timesteps/leak, current timesteps/leak {}/{}, loaded timesteps/leak {}/{}'.format(timesteps, leak, state['timesteps'], state['leak']))
            thresholds = state['thresholds']
            model.module.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])
        else:
            f.write('\n Loaded SNN model does not have thresholds')

    f.write('\n {}'.format(model))
    
    max_accuracy = 0
    avg_spike_dict ={}

    start_time = datetime.datetime.now()

    evaluate_sample(1)
    # avg_spikerates = inference(test_loader)