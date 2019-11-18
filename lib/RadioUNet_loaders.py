from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

import warnings
warnings.filterwarnings("ignore")


class RadioUNet_dataset_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds,ind1,ind2, csv_file, root_dir_gain, root_dir_buildings, root_dir_antenna, transform=None):
        """
        Args:
            maps_inds: an np array of indeces of mpas, used for shuffling
            ind1: the lowest map index to load
            ind2: the highest map index to load
            csv_file (string): Path to the csv file with annotations.
            root_dir_gain (string): Directory with all the gain images.
            root_dir_buildings (string): Directory with all the building images.
            root_dir_antenna (string): Directory with all the antenna images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ind1=ind1
        self.ind2=ind2
        self.maps_inds=maps_inds
        self.gain_frame = pd.read_csv(csv_file)
        self.root_dir_gain = root_dir_gain
        self.root_dir_buildings = root_dir_buildings
        self.root_dir_antenna = root_dir_antenna
        self.transform = transform
        self.height = 256
        self.width = 256
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*100
    
    def __getitem__(self, idx):
        idxr=np.floor(idx/100).astype(int)
        idxc=idx-idxr*100
        img_name_buildings = os.path.join(self.root_dir_buildings,
                                          self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 2])
        img_name_antenna = os.path.join(self.root_dir_antenna,
                                        self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 7 +200 + idxc])
        img_name_gain = os.path.join(self.root_dir_gain,
                                     self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 7+idxc])        
        image_buildings = np.asarray(io.imread(img_name_buildings))
        image_antenna = np.asarray(io.imread(img_name_antenna))
        image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)
        
                
        image_build_ant=np.stack([image_buildings, image_antenna], axis=2) #note that ToTensor moves the channel
                                                                           #from the last asix to the first!

        
        if self.transform:
            image_build_ant = self.transform(image_build_ant)
            image_gain = self.transform(image_gain)


        return [image_build_ant, image_gain]
    
    
    
    
    
    
    
    
    
class RadioUNet_dataset_s(Dataset):
    """RadioMapSeer Loader for maps with a missing building and measurements (RadioUNet_s)"""
    def __init__(self,maps_inds,ind1,ind2, csv_file, root_dir_gain, root_dir_buildings_miss,
                 root_dir_buildings_all, root_dir_antenna, transform=None,miss=0, fix_samples=0,
                 num_samples_low= 10, num_samples_high= 300):
        """
        Args:
            maps_inds: an np array of indeces of mpas, used for shuffling
            ind1: the lowest map index to load
            ind2: the highest map index to load
            csv_file (string): Path to the csv file with annotations.
            root_dir_gain (string): Directory with all the gain images.
            root_dir_buildings_miss (string): Directory with the building images, missing one building.
            root_dir_buildings_all (string): Directory with all the building images.
            root_dir_antenna (string): Directory with all the antenna images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            miss: if zero, return all buildings, otherwise return a image with a missing building
            fix_samples: if zero, take a random number of samples between num_samples_low and num_samples_high. If nonzero, the
            number of samples is fix_samples 
            num_samples_low: lowest number of samples, if the number of samples is random
            num_samples_hig: lhighest number of samples, if the number of samples is random
        """
        self.ind1=ind1
        self.ind2=ind2
        self.maps_inds=maps_inds
        self.gain_frame = pd.read_csv(csv_file)
        self.root_dir_gain = root_dir_gain
        self.root_dir_buildings_miss = root_dir_buildings_miss
        self.root_dir_buildings_all = root_dir_buildings_all
        self.root_dir_antenna = root_dir_antenna
        self.transform = transform
        self.height = 256
        self.width = 256
        self.miss=miss
        self.fix_samples=fix_samples
        self.num_samples_low=num_samples_low
        self.num_samples_high=num_samples_high
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*100
    
    def __getitem__(self, idx):
        idxr=np.floor(idx/100).astype(int)
        idxc=idx-idxr*100
        if (self.miss==0):
            img_name_buildings = os.path.join(self.root_dir_buildings_all,
                                          self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 2])
        else:
            img_name_buildings = os.path.join(self.root_dir_buildings_miss,
                                          self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 0])
        
        img_name_antenna = os.path.join(self.root_dir_antenna,
                                        self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 7 +200 + idxc])
        img_name_gain = os.path.join(self.root_dir_gain,
                                     self.gain_frame.iloc[self.maps_inds[idxr+self.ind1]+1, 7+idxc])        
        image_buildings = np.asarray(io.imread(img_name_buildings))/256
        image_antenna = np.asarray(io.imread(img_name_antenna))/256
        image_samples = np.zeros((256,256))
        image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)
        
        if self.fix_samples==0:
            num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_samples=np.floor(self.fix_samples).astype(int)
                
        x_samples=np.random.randint(0, 255, size=num_samples)
        y_samples=np.random.randint(0, 255, size=num_samples)
        image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]/256
     
        image_build_ant_samples=np.stack([image_buildings, image_antenna, image_samples], axis=2) #note that ToTensor moves the channel
                                                                                                  #from the last asix to the first!
        
        if self.transform:
            image_build_ant_samples = self.transform(image_build_ant_samples).float()
            image_gain = self.transform(image_gain).float()

        return [image_build_ant_samples, image_gain]