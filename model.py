# created by Jonathan Jeremie Randriarison at 20230206 22:13.
# 
# Multi (Variationnal)Autoencoder for fault detection and isolation
# Fault explainability 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import train
from dataset import create_dataset
from cost_function import criterion

# Define our experimental model , Autoencoder , Variational Autoencoder and multi-V(autoencoder)
class Autoencoder(nn.Module) :
    def __init__(self):
        super().__init__()
        size_in = 46
        size_hidden_1 = 54
        size_hidden_2 = 78
        size_hidden_3 = 96
        size_latent_space = 92
        dropout_proba = 0.0
        self.encoder = nn.Sequential(
             nn.Linear(size_in , size_hidden_1 ) , # N,6 -> N,5
             nn.ReLU() ,
             #nn.LeakyReLU(0.1) ,
             nn.Dropout(dropout_proba) ,
             nn.Linear(size_hidden_1, size_hidden_2) ,
             nn.ReLU(),
             #nn.LeakyReLU(0.1) ,
             nn.Dropout(dropout_proba) ,
             nn.Linear(size_hidden_2,size_latent_space) # --> 2
             )
        
        self.decoder = nn.Sequential(
             nn.Linear(size_latent_space, size_hidden_2) , # N,2 -> N,3
             nn.ReLU() ,
             nn.Dropout(dropout_proba) ,
             nn.Linear(size_hidden_2, size_hidden_1) ,
             nn.ReLU() ,
             nn.Dropout(dropout_proba) ,
             nn.Linear(size_hidden_1, size_in), # --> N, 6
             )
    
    def forward(self, x) :
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded , z

class VariationalAutoEncoder(nn.Module) :
    def __init__(self):
        size_in = 46
        size_hidden_1 = 54
        size_hidden_2 = 78
        size_hidden_3 = 86
        size_latent_space = 92
        super().__init__()
        self.encoder = nn.Sequential(
             nn.Linear(size_in, size_hidden_1 ) , 
             nn.ReLU() ,
             nn.Linear(size_hidden_1, size_hidden_2) ,
             nn.ReLU(),
             nn.Linear(size_hidden_2,size_latent_space) 
             )
        
        self.linear2 = nn.Linear(size_latent_space, size_latent_space)
        self.linear3 = nn.Linear(size_latent_space, size_latent_space)
        
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        
        self.decoder = nn.Sequential(
             nn.Linear(size_latent_space, size_hidden_2) ,
             nn.ReLU() ,
             nn.Linear(size_hidden_2, size_hidden_1) ,
             nn.ReLU() ,
             nn.Linear(size_hidden_1,size_in), 
             )
    
    def forward(self, x) :
        x = self.encoder(x)
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        decoded = self.decoder(z)
        return decoded , z    


class Multi_ae :
    def __init__(self,nb_fault_class,mode="AE", train_one_new_class=True,device="cpu"):
        self.mode = mode
        self.models = []
        self.nb_fault_class = nb_fault_class
        self.unknown_class_data = []
        self.unknown_data_number = 0
        self.unknown_data = None
        
        self.minmax_rloss_train_data = []
        self.normalizing = False
        
        self.train_one_new_class = train_one_new_class
        self.device = device

        for _ in range(nb_fault_class) :
            if self.mode == "AE":
                model = Autoencoder().double().to(device)
                self.models.append(model)
            elif self.mode == "VAE":
                model = VariationalAutoEncoder().double().to(device)
                self.models.append(model)
    
    def load(self,model_path,model_name):
        saved_model = torch.load(model_path)
        for idx ,name in enumerate(model_name):
            self.models[idx].load_state_dict(saved_model[name])
        
    def train(self, data_loaders : list , n_epochs):
        for num_model in range(self.nb_fault_class):
            self.models[num_model] = train(data_loaders[num_model],self.models[num_model],model_type=self.mode,model_num=num_model,num_epochs=n_epochs)
    
    def compute_recon_minmax(self,train_data,new_model=False):
        data_loader = create_dataset(train_data,batch_size=1)
        if not new_model :
            for num_model in range(self.nb_fault_class):
                r_loss = [criterion(self.models[num_model](data_point)[0],data_point,self.models[num_model],model_type=self.mode).detach().numpy().item()  for data_point, _ in data_loader ]
                self.minmax_rloss_train_data.append((min(r_loss),max(r_loss)))
        else :
            r_loss = [criterion(self.models[-1](data_point)[0],data_point,self.models[-1],model_type=self.mode).detach().numpy().item()  for data_point, _ in data_loader ]
            self.minmax_rloss_train_data.append((min(r_loss),max(r_loss)))
    
    def predict(self, data_point, threshold = 1, detect_new_class = False):
        r_loss = []
        
        if self.unknown_data_number >= 50 and detect_new_class:
            self.train_new_ae(20)
            self.unknown_data_number = 0
            #self.unknown_class_data = []
        
        # normalize ( if normalizing variable is True ) reconstruction loss with min and max of training data 
        for num_model in range(len(self.models)) :
            model = self.models[num_model]
            loss = criterion(model(data_point)[0],data_point,model,model_type=self.mode).detach().numpy().item()
            if len(self.minmax_rloss_train_data) > 0 and self.normalizing :
                max_loss = self.minmax_rloss_train_data[num_model][1]
                min_loss = self.minmax_rloss_train_data[num_model][0]
                minmax_func = lambda loss : (loss - min_loss) / (max_loss - min_loss)
                r_loss.append(minmax_func(loss))
            else :
                r_loss.append(loss)
        
        # get the expected class by argmin of reconstruction loss and detect new class ( if detect_new_class is True ) 
        if all(r_l > threshold for r_l in r_loss[0:self.nb_fault_class]) and detect_new_class :
            new_class_num = torch.tensor([self.nb_fault_class])
            d_point = torch.cat((data_point[0], new_class_num ), 0).detach().numpy().tolist()
            self.unknown_class_data.append(d_point)
            self.unknown_data_number+=1
            
            while len(r_loss) != 4 :
                    r_loss.append(float('inf'))
                    
            return self.nb_fault_class , r_loss
        else : 
            if not detect_new_class :
                return np.argmin(np.array(r_loss)).item() , r_loss
            else : 
                while len(r_loss) != 4 :
                    r_loss.append(float('inf'))
                if all(r_l > threshold for r_l in r_loss[0:self.nb_fault_class]) :
                    return self.nb_fault_class , r_loss
                else :
                    return np.argmin(np.array(r_loss[0:self.nb_fault_class])).item() , r_loss
    
    def train_new_ae(self,n_epochs):
        device = self.device
        col_name = []
        
        self.unknown_data = pd.DataFrame(self.unknown_class_data , columns=col_name)
        data_loader = create_dataset(self.unknown_data , batch_size=32)
        model_num = self.nb_fault_class
        if  self.train_one_new_class :
            if self.mode == "AE":
                model = Autoencoder().double().to(device)
            elif self.mode == "VAE":
                model = VariationalAutoEncoder().double().to(device)
            self.models.append(model)
            self.train_one_new_class = False
            #self.nb_fault_class+=1
        
        self.models[-1] = train(data_loader,self.models[-1],model_type=self.mode,model_num = 3 , num_epochs= n_epochs)
        #self.nb_fault_class+=1
        
        if self.normalizing :
            self.compute_recon_minmax(self.unknown_data,new_model=True)