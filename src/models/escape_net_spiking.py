#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

global spike_dict

class PoissonGenerator(nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*0.9).float(),torch.sign(input))
		return out

class STDB(torch.autograd.Function):

	alpha 	= ''
	beta 	= ''
    
	@staticmethod
	def forward(ctx, input, last_spike):
        
		ctx.save_for_backward(last_spike)
		out = torch.zeros_like(input)
		out[input > 0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):
	    		
		last_spike, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		return grad*grad_input, None

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None


class ESCAPE_NET_SNN_STDB(nn.Module):
  def __init__(self, model_name, activation='Linear', labels=3, timesteps=100, leak=1.0, default_threshold = 1.0, alpha=0.3, beta=0.01, dropout=0.2, kernel_size=8, dataset='RAT4'):
    super().__init__()
    cfg = {
      'IG1': [64, 'A'],
      'IG2': [64, 'A', 64, 'A'],
      'ESCAPE_NET': [64, 'A', 64, 'A', 64]
    }
    self.model_name 	= model_name

    if activation == 'Linear':
      self.act_func 	= LinearSpike.apply
    elif activation == 'STDB':
      self.act_func	= STDB.apply

    self.labels 		= labels
    self.timesteps 	= timesteps
    self.leak 	 		= torch.tensor(leak)
    STDB.alpha 		 	= alpha
    STDB.beta 			= beta 
    self.dropout 		= dropout
    self.kernel_size 	= kernel_size
    self.dataset 		= dataset
    self.input_layer 	= PoissonGenerator()
    self.threshold 		= {}
    self.mem 			= {}
    self.mask 			 = {}
    self.spike 			 = {}
    self.avg_spike_dict = {}
    self.conv_layers = []
    self.fc_layers   = []

    self.features, self.classifier = self._make_layers(cfg['ESCAPE_NET'])

    self._initialize_weights2()

    for i, _ in enumerate(self.features):
      if isinstance(self.features[i], nn.Conv2d):
        self.conv_layers.append(i)
        self.avg_spike_dict[i] = []
        self.threshold[i] 	= torch.tensor(default_threshold)
        
    prev = len(self.features)
    for i, _ in enumerate(self.classifier):
      if isinstance(self.classifier[i], nn.Linear):
        self.fc_layers.append(prev+i)
        self.avg_spike_dict[prev+i] = []
        self.threshold[prev+i] 	= torch.tensor(default_threshold)

  def _initialize_weights2(self):
    for m in self.modules():
            
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
          m.bias.data.zero_()

  def threshold_update(self, scaling_factor=1.0, thresholds=[]):
    # Initialize thresholds
    self.scaling_factor = scaling_factor

    for pos in range(len(self.features)):
      if isinstance(self.features[pos], nn.Conv2d):
        if thresholds:
          self.threshold[pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
        #print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))

    prev = len(self.features)

    for pos in range(len(self.classifier)-1):
      if isinstance(self.classifier[pos], nn.Linear):
        if thresholds:
          self.threshold[prev+pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
        #print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))

  def _make_layers(self, cfg):
    # Defining the feature extraction layers
    layers 		= []
    in_channels = 1
    padding = 'same'

    if self.model_name == 'ESCAPE_NET':
      pool_size = [2, 2, 2]
    count = 0

    for x in (cfg):
      stride = 1
            
      if x == 'A':
        layers += [nn.AvgPool2d(kernel_size=pool_size[count])]
        count+=1
      else:
        layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding='same', stride=stride, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        # layers += [nn.Dropout(self.dropout)]
        in_channels = x
        self.kernel_size = self.kernel_size//2

    features = nn.Sequential(*layers)
    feature_layers = len(layers)
    
    # Defining the classification layers
    layers = []
    if self.model_name == 'ESCAPE_NET':
      layers += [nn.Linear(64*14*25, 256, bias = False)]
      layers += [nn.ReLU(inplace=True)]
      # layers += [nn.Dropout(0.5)]
      layers += [nn.Linear(256, 3, bias = False)]
      layers += [nn.Softmax(dim=1)]

    classifer = nn.Sequential(*layers)
    return (features, classifer)

  def network_update(self, timesteps, leak):
    self.timesteps 	= timesteps
    self.leak 	 	= torch.tensor(leak)
	
  def neuron_init(self, x):
    self.batch_size = x.size(0)
    self.width 		= x.size(2)
    self.height 	= x.size(3)

    self.mem 		= {}
    self.mask 		= {}
    self.spike 		= {}			
				
    for l in range(len(self.features)):
                
      if isinstance(self.features[l], nn.Conv2d):
        self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
      
      elif isinstance(self.features[l], nn.Dropout):
        self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape))

      elif isinstance(self.features[l], nn.AvgPool2d):
        self.width = self.width//self.features[l].kernel_size
        self.height = self.height//self.features[l].kernel_size
    
    prev = len(self.features)

    for l in range(len(self.classifier)):
      
      if isinstance(self.classifier[l], nn.Linear):
        self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features)
      
      elif isinstance(self.classifier[l], nn.Dropout):
        self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape))
        
    self.spike = copy.deepcopy(self.mem)
    for key, values in self.spike.items():
      for value in values:
        value.fill_(-1000)

  def forward(self, x, find_max_mem=False, max_mem_layer=0):
    global layers
    self.neuron_init(x)
    max_mem=0.0

    for t in range(self.timesteps):
      out_prev = self.input_layer(x)
      
      for l in range(len(self.features)):
        
        if isinstance(self.features[l], (nn.Conv2d)):
          
          if find_max_mem and l==max_mem_layer:
            if (self.features[l](out_prev)).max()>max_mem:
              max_mem = (self.features[l](out_prev)).max()
            break

          mem_thr 		= (self.mem[l]/self.threshold[l]) - 1.0
          out 			= self.act_func(mem_thr, (t-1-self.spike[l]))
          rst 			= self.threshold[l]* (mem_thr>0).float()
          self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1)
          self.mem[l] 	= (self.leak*self.mem[l] + self.features[l](out_prev) - rst)
          out_prev  		= out.clone()

        elif isinstance(self.features[l], nn.AvgPool2d):
          out_prev 		= self.features[l](out_prev)
        
        elif isinstance(self.features[l], nn.Dropout):
          out_prev 		= out_prev * self.mask[l]
      
      if find_max_mem and max_mem_layer<len(self.features):
        continue

      out_prev       	= out_prev.reshape(self.batch_size, -1)
      prev = len(self.features)
      
      for l in range(len(self.classifier)-1):
                          
        if isinstance(self.classifier[l], (nn.Linear)):
          
          if find_max_mem and (prev+l)==max_mem_layer:
            if (self.classifier[l](out_prev)).max()>max_mem:
              max_mem = (self.classifier[l](out_prev)).max()
            break

          mem_thr 			= (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
          out 				= self.act_func(mem_thr, (t-1-self.spike[prev+l]))
          rst 				= self.threshold[prev+l] * (mem_thr>0).float()
          self.spike[prev+l] 	= self.spike[prev+l].masked_fill(out.bool(),t-1)
          self.mem[prev+l] 	= (self.leak*self.mem[prev+l] + self.classifier[l](out_prev) - rst)
          out_prev  		= out.clone()

        elif isinstance(self.classifier[l], nn.Dropout):
          out_prev 		= out_prev * self.mask[prev+l]
      
      # Compute the classification layer outputs
      if not find_max_mem:
        prev-=1
        self.mem[prev+l+1] 	= self.mem[prev+l+1] + self.classifier[l+1](out_prev)
    if find_max_mem:
      return max_mem
    self.count_spikes() #the number of spikes in a given layer averaged over the current batch 

    return self.mem[prev+l+1]

  def count_spikes(self):
    res = 0
    for layer in self.conv_layers: #convolutional layers
      res = 0
      torch.nn.functional.relu(self.spike[layer], inplace=True)
      for i, _ in enumerate(self.spike[layer]):
        res += torch.sum(self.spike[layer][i])
      res/=len(self.spike[layer])
      # print(res)
      self.avg_spike_dict[layer].append(res.item())

    for layer in self.fc_layers: #linear layers
      res = 0
      torch.nn.functional.relu(self.spike[layer], inplace=True)
      for i, _ in enumerate(self.spike[layer]):
        res += torch.sum(self.spike[layer][i])
      res/=len(self.spike[layer])
      # print(res)
      self.avg_spike_dict[layer].append(res.item())

def test():
      net = ESCAPE_NET_SNN_STDB('ESCAPE_NET')
      x = torch.randn(1,1,56,100)
      net = nn.DataParallel(net) 
      if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net.cuda()
        x = x.cuda()
      y = net(x)
      print(net.module.avg_spike_dict)
      # print(y.size())

if __name__ == '__main__':
    test()
    # test_model = ESCAPE_NET_SNN_STDB(model_name='ESCAPE_NET', labels=3, dataset='RAT4', kernel_size=8, dropout=0.2)
    # print(test_model)