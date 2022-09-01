import argparse
import os
import pdb
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

from models import escape_net as ANN
from data import make_dataset as Processor
from utils import CustomDataset, AverageMeter

def train(epoch, loader):

    global learning_rate
    
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')


    #total_correct   = 0
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        start_time = datetime.datetime.now()

        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        #make_dot(loss).view()
        #exit(0)
        loss.backward()
        optimizer.step()
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        #total_correct += correct.item()

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
        
    f.write(
        '\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
        epoch,
        learning_rate,
        losses.avg,
        top1.avg
        )
    )
    history.setdefault('train_loss', []).append(losses.avg)
    history.setdefault('train_acc', []).append(top1.avg)

def test(loader):
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        global max_accuracy, start_time
        
        for batch_idx, (data, target) in enumerate(loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(), data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))

        if epoch>30 and top1.avg<0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
            state = {
                    'accuracy'      : max_accuracy,
                    'epoch'         : epoch,
                    'state_dict'    : model.state_dict(),
                    'optimizer'     : optimizer.state_dict()
            }
            filename = os.path.join(save_path, save_name)
            torch.save(state,filename)
            
        f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.  format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )
        history.setdefault('test_loss', []).append(losses.avg)
        history.setdefault('test_acc', []).append(top1.avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', default=True,               type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=120,                type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='RAT4',             type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='IG2',              type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=0.001,              type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--epochs',                 default=150,                 type=int,       help='number of training epochs')
    parser.add_argument('--optimizer',              default='Adam',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=1e-6,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.9,                type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--dataset_path',           default='D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/data/processed/Rat4Training_Fold1.mat', type=str, help = '')
    parser.add_argument('--save_dir',               default = '',               tpye=str,       help='')
    parser.add_argument('--save_name',              default = '',               tpye=str,       help='')
    # parser.add_argument('-f')
    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(240)
    np.random.seed(240)
    torch.cuda.manual_seed(240)
    torch.cuda.manual_seed_all(240)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    dataset         = args.dataset
    batch_size      = args.batch_size
    learning_rate   = args.learning_rate
    pretrained_ann  = args.pretrained_ann
    architecture    = args.architecture
    epochs          = args.epochs
    optimizer       = args.optimizer
    weight_decay    = args.weight_decay
    momentum        = args.momentum
    amsgrad         = args.amsgrad
    dropout         = args.dropout
    kernel_size     = args.kernel_size
    dataset_path    = args.dataset_path
    # save_dir            = args.save_dir
    save_dir            = 'D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/artifacts/train/trained_models/ann'
    # save_name           = args.save_name
    save_name_base           = 'ann_escape_net_ann_accuracy_timesteps_number'
    
    
    save_path = os.path.join(save_dir, save_name_base)
    try:
        os.mkdir(save_path)
    except OSError:
        #handle this exception better
        pass 

    log_file = save_name_base + '.log'
    log_file = os.path.join(save_path, log_file)
    
    if args.log:
        f= open(log_file, 'w', buffering=1)
    else:
        f=sys.stdout
    
    
    f.write('\n Run on time: {}'.format(datetime.datetime.now()))
            
    f.write('\n\n Arguments:')
    for arg in vars(args):
        f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
        
    # Training settings
    '''
    separation of concerns
    '''
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Loading Dataset
    if dataset == 'RAT4':
        labels = 3
        RAT_data = Processor.Preprocessing_module('Rat4',1,dataset_path)
        train_dataset   = CustomImageDataset(RAT_data.training_set, RAT_data.training_labels, transforms.ToTensor())
        test_dataset    = CustomImageDataset(RAT_data.test_set, RAT_data.test_labels, transforms.ToTensor())
    
    if torch.cuda.is_available() and args.gpu:
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    for architecture in ['IG1', 'IG2', 'ESCAPE_NET']:
        if architecture == 'IG1':
            model = ANN.ESCAPE_NET(model_name='IG1', labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
            model = nn.DataParallel(model) 
            f.write('\n {}'.format(model))

            save_name = save_name_base + '_ig1.pth'
            history_save_path = os.path.join(save_path, 'history_ig1.npy')

        elif architecture == 'IG2':
            pretrained_ig1 = os.path.join(save_path, save_name)
            pretrained_model = ANN.ESCAPE_NET(model_name='IG1', labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
            pretrained_model = nn.DataParallel(pretrained_model) 
            pretrained_model.load_state_dict(torch.load(pretrained_ig1))
            
            model = ANN.ESCAPE_NET(model_name='IG2', labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
            model = nn.DataParallel(model) 

            pretrained_dict = pretrained_model.state_dict()
            cur_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cur_dict and 'module.classifier' not in k}
            cur_dict.update(pretrained_dict) 
            model.load_state_dict(cur_dict)
            f.write('\n {}'.format(model))

            save_name = save_name_base + '_ig2.pth'
            history_save_path = os.path.join(save_path, 'history_ig2.npy')
            
        elif architecture == 'ESCAPE_NET':
            pretrained_ig2 = os.path.join(save_path, save_name)
            pretrained_model = ANN.ESCAPE_NET(model_name='IG2', labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
            pretrained_model = nn.DataParallel(pretrained_model) 
            pretrained_model.load_state_dict(torch.load(pretrained_ig2))
            
            model = ANN.ESCAPE_NET(model_name='ESCAPE_NET', labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
            model = nn.DataParallel(model) 

            pretrained_dict = pretrained_model.state_dict()
            cur_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cur_dict and 'module.classifier' not in k}
            cur_dict.update(pretrained_dict) 
            model.load_state_dict(cur_dict)
            f.write('\n {}'.format(model))

            save_name = save_name_base + '_escape_net.pth'
            history_save_path = os.path.join(save_path, 'history.npy')
        
        if torch.cuda.is_available() and args.gpu:
            model.cuda()
        
        if optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov = True)
        elif optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), capturable=True)
        
        f.write('\n {}'.format(optimizer))
        max_accuracy = 0
        history = {}
        for epoch in range(1, epochs):
            start_time = datetime.datetime.now()
            train(epoch, train_loader)
            test(test_loader)
            np.save(history_save_path, history)
              
        f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))