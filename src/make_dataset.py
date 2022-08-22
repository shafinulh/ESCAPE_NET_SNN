# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:24:24 2018

@author: haques (adopted from kohr)
"""

import numpy as np 
import scipy.io
import random

class Preprocessing_module:
    
    '''Constructor'''    
    def __init__(self,Ratnum,foldnum,dataset_path):
    
        RAT = scipy.io.loadmat(dataset_path)
        
        training_data = RAT['training_data_rat'];
        test_data = RAT['test_data_rat'];
        
        training_data = training_data.T;
        test_data = test_data.T;
        
        test_data = test_data.reshape(test_data.shape[0],56,100,1)
        training_data = training_data.reshape(training_data.shape[0],56,100,1)
        
        self.training_set = training_data
        self.training_labels = RAT['training_data_labels']
        self.test_set = test_data
        self.test_labels = RAT['test_data_labels']
        self.valid_set = []
        self.valid_labels = []
        self.samples1 = []
        self.samples2 = []
        self.samples3 = []
        
    def Reshape_data_set(self,training_data,test_data):
        
        self.test_set = test_data.reshape(test_data.shape[0],56,100,1)
        self.training_set = training_data.reshape(training_data.shape[0],56,100,1)
        
    def Randomize_Train_and_getValidSet(self,training_data,training_labels,numspikes):
        itemindex = np.where(training_labels==1)
        temp = random.sample(range(0,itemindex[0].shape[0]),numspikes)
        self.samples1 = temp
        validation_set = training_data[itemindex[0][temp],:,:,:]
        validation_labels =  training_labels[itemindex[0][temp]]
        indices = itemindex[0][temp]
        itemindex = np.where(training_labels==2)
        temp = random.sample(range(0,itemindex[0].shape[0]),numspikes)
        self.samples2 = temp
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        itemindex = np.where(training_labels==3)
        temp = random.sample(range(0,itemindex[0].shape[0]),numspikes)
        self.samples3 = temp
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        
        training_data = np.delete(training_data,indices,axis = 0)
        training_labels = np.delete(training_labels,indices,axis = 0)
        
        self.valid_set = validation_set
        self.valid_labels = validation_labels
        self.training_set = training_data
        self.training_labels = training_labels
        
    def Randomize_Train_and_getValidSet2(self,training_data,training_labels,sample1,sample2,sample3,numspikes):
        itemindex = np.where(training_labels==1)
        temp = sample1
        validation_set = training_data[itemindex[0][temp],:,:,:]
        validation_labels =  training_labels[itemindex[0][temp]]
        indices = itemindex[0][temp]
        itemindex = np.where(training_labels==2)
        temp = sample2
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        itemindex = np.where(training_labels==3)
        temp = sample3
        validation_set = np.concatenate((validation_set,training_data[itemindex[0][temp],:,:,:]), axis = 0);
        validation_labels = np.concatenate((validation_labels,training_labels[itemindex[0][temp]]));
        indices = np.concatenate((indices, itemindex[0][temp]))
        
        training_data = np.delete(training_data,indices,axis = 0)
        training_labels = np.delete(training_labels,indices,axis = 0)
        
        self.valid_set = validation_set
        self.valid_labels = validation_labels
        self.training_set = training_data
        self.training_labels = training_labels