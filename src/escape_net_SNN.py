#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
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
from sklearn import model_selection

import make_dataset
from models import escape_net_spiking as SNN

class CustomDataset(Dataset):
    """Modifying torch built-in dataset class to be capable of handling the RAT dataset"""
    def __init__(self, data_all, labels, transform=None, target_transform=None):
        self.labels = labels
        self.data_all = data_all
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data_all[idx]
        label = self.labels[idx][0]-1
        if self.transform:
            data = self.transform(data)
            data = data.float()
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def find_threshold(batch_size=512, timesteps=2500, architecture='ESCAPE_NET'):

    loader = train_loader
    try:
        obj = model.module
    except AttributeError:
        obj = model
    
    obj.network_update(timesteps=timesteps, leak=1.0)
    

    pos=0
    thresholds=[]
    
    def find(layer, pos):
        max_act=0
        
        f.write('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()

                #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    pos = pos+1
                    f.write(' {}'.format(thresholds))
                    obj.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
        return pos

    if architecture=='ESCAPE_NET':              
        for l in obj.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
        
        for c in obj.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if (int(l[0])+int(c[0])+1) == (len(obj.features) + len(obj.classifier) -1):
                    pass
                else:
                    pos = find(int(l[0])+int(c[0])+1, pos)

    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):
    global learning_rate
    global history

    model.module.network_update(timesteps=timesteps, leak=leak)
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    model.train()
       
    current_time = start_time

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data) 
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()        
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(),data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))

        if (batch_idx+1) % train_acc_batches == 0:
            temp1 = []
            for value in model.module.threshold.values():
                temp1 = temp1+[round(value.item(),2)]
            f.write('\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}'
                    .format(epoch,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        model.module.leak.item(),
                        model.module.timesteps
                        )
                    )
            
    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'
                    .format(epoch,
                        learning_rate,
                        losses.avg,
                        top1.avg,
                        )
                    )
    history.setdefault('train_loss', []).append(losses.avg)
    history.setdefault('train_acc', []).append(top1.avg)
      
def test(epoch):
    global history

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output  = model(data) 
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
        
        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1+[value.item()]    

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : leak,
                    'activation'            : activation
                  }
            filename = save_name+'.pth'
            filename = os.path.join(save_path, filename)
            torch.save(state,filename)    

        f.write('\n test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )
        history.setdefault('test_loss', []).append(losses.avg)
        history.setdefault('test_acc', []).append(top1.avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='RAT4',             type=str,       help='dataset name')
    parser.add_argument('--batch_size',             default=32,                 type=int,       help='minibatch size')
    parser.add_argument('--architecture',           default='ESCAPE_NET',       type=str,       help='network architecture')
    parser.add_argument('-lr',                      default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--log',                    default=True,             type=bool,      help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=10,                 type=int,       help='number of training epochs')
    parser.add_argument('--timesteps',              default=28,                 type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=1.0,                type=float,     help='scaling factor for thresholds at reduced timesteps')
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
    parser.add_argument('--save_dir',               default = '',               type=str,       help='')
    parser.add_argument('--save_name',              default = '',               type=str,       help='')
    # parser.add_argument('-f')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = 'ESCAPE_NET'
    learning_rate       = args.lr
    pretrained_ann      = args.pretrained_ann
    # pretrained_snn      = args.pretrained_snn
    pretrained_snn      = 'D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/trained_models/snn/snn_escape_net_727_28_1.pth'
    epochs              = args.epochs
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
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
    dataset_path        = 'D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/data/processed/Rat4Training_Fold1.mat'
    # save_dir            = args.save_dir
    save_dir            = 'D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/trained_models/snn'
    # save_name           = args.save_name
    save_name           = 'snn_escape_net_ann_accuracy_timesteps_number'

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
        if arg == 'pretrained_ann':
            f.write('\n\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    
    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'RAT4':
        labels = 3
        RAT_data = make_dataset.Preprocessing_module('Rat4',1,dataset_path)
        train_dataset   = CustomDataset(RAT_data.training_set, RAT_data.training_labels, transforms.ToTensor())
        test_dataset    = CustomDataset(RAT_data.test_set, RAT_data.test_labels, transforms.ToTensor())
    
    if torch.cuda.is_available() and args.gpu:
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


    model = SNN.ESCAPE_NET_SNN_STDB(model_name='ESCAPE_NET', activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, kernel_size=kernel_size, dataset=dataset)
    model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()     
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        model.load_state_dict(cur_dict)
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        if 'thresholds' not in state.keys() or len(state['thresholds'])==0:
            thresholds = find_threshold(batch_size=512, timesteps=1000, architecture=architecture)
            try:
                model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            except AttributeError:
                model.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            
            #Save the threhsolds in the ANN file
            temp = {}
            for key,value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, pretrained_ann)
        else:
            thresholds = state['thresholds']
            f.write('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            try :
                model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            except AttributeError:
                model.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])

    elif pretrained_snn:  
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
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
        else:
            f.write('\n Loaded SNN model does not have thresholds')

    f.write('\n {}'.format(model))
    f.write('\n Thresholds: {}'.format(state['thresholds']))
    f.write('\n Thresholds scaled down: {}'.format(model.module.threshold))

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, capturable = False)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
    f.write('\n {}'.format(optimizer))
    
    max_accuracy = 0
    history = {}
    history_save_path = os.path.join(save_path, 'history.npy')

    for epoch in range(1, epochs):
        start_time = datetime.datetime.now()
        train(epoch)
        test(epoch)
        np.save(history_save_path, history)
    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))
  