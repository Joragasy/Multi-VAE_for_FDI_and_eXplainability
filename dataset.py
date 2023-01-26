# created by Jonathan Jeremie Randriarison at 20230206 22:13.
# 
# Multi (Variationnal)Autoencoder for fault detection and isolation
# Fault explainability

import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader,Dataset

# this class allow us to prepare data to be suitable for our model
class Dataset(Dataset):
    
    def __init__(self, data ):
        training_data = data
        training_data = training_data.astype(np.float32)
        self.training_data = training_data
        
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self,index):
        elem = self.training_data.iloc[index][1:-1]
        elem = np.array(list(elem),dtype=np.float64)
        target = self.training_data.iloc[index][-1]
        return elem , target
    
# create data loader for batch creation , data loader take as input the output of Dataset created above
def create_dataset(data , batch_size = 64):
    data_builder = Dataset(data)
    data_loader = DataLoader(data_builder,
                             shuffle=False,
                             batch_size=batch_size,
                             pin_memory=True)
    return data_loader