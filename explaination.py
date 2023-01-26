# created by Jonathan Jeremie Randriarison at 20230206 22:13.
# 
# Multi (Variationnal)Autoencoder for fault detection and isolation
# Fault explainability

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import torch
from cost_function import criterion
from dataset import create_dataset

# explain only one occurence of fault
def local_explaination(X,model):
    data_loader = create_dataset(X.to_frame().T,batch_size=1)
    for in_data , _ in data_loader :
        x = in_data
    # with requires_grad=True we tell to pytorch that we need the partial derivative of the input to make the input learnable
    x = torch.nn.Parameter(x, requires_grad=True) 
    # we want to optimize only the input 
    optim = torch.optim.Adam([x], lr=1e-3) 
    
    # iteration limit is useful to break learning phase when the value of loss function doesn't reach the predefined threshold 
    iteration_limit = 2000
    thresold = 0.05
    normal_model = model
    while True :
        # we use the autoencoder who encode normal data for explaination
        x_hat = normal_model(x)[0]
        loss = criterion(x,x_hat,normal_model,model_type="VAE")
        optim.zero_grad()
        # backpropagation step => it compute all partial derivative of Loss function by input from all wieghts and bias
        loss.backward() 
        if abs(loss) < thresold or iteration_limit == 0 :
            break
        # optimization step using Adam optimizer defined at the top
        optim.step() 
        iteration_limit-=1
    
    res = x[0].detach().numpy().tolist()
    return res


def show_local_explaination(list_to_explain,
                      original_data_point,
                      invert_transform,
                      data_train,
                      impact_pourcentage = 50 ,
                      plot=False,
                      sort=False,
                      show_pourcentage=False):
    exp_data = pd.DataFrame(list_to_explain)
    exp_data = exp_data.T
    exp_data.columns = data_train.columns[1:-1]
    col_to_scale = list(data_train.columns)[1:-1]
    if invert_transform :
        scaler = MinMaxScaler()
        scaler.fit(data_train[col_to_scale])
        exp_inverted = pd.DataFrame()
        org_inverted = pd.DataFrame()
        exp_inverted[col_to_scale] = scaler.inverse_transform(exp_data[col_to_scale])
        org_inverted[col_to_scale] = scaler.inverse_transform(original_data_point[col_to_scale])

        result = pd.concat([org_inverted,exp_inverted]).T
    else :
        result = pd.concat([original_data_point[col_to_scale],exp_data[col_to_scale]]).T
        
    result.columns = ['real_data_point','optimal_values']
    result['impact'] = result[result.columns].apply(lambda x  : x[0] - x[1] , axis=1)
    result = result.fillna(0)
    #full_deviation = result.impact.abs().sum()
    result['pourcentage_impact'] = result['impact'].apply(lambda x : (abs(x)/result.impact.abs().sum())*100 )
    
    result_copy = result.sort_values(by=['pourcentage_impact'],ascending=False)
    result_copy['pourc_cum'] = result_copy.pourcentage_impact.cumsum()
    impacting_variable_index = list(np.where(result_copy['pourc_cum'] > impact_pourcentage))[0][0]
    impacting_variable_line = (len(result) - impacting_variable_index ) - 1.5
    
    if sort :
            result.sort_values(by=['pourcentage_impact'],ascending=True , inplace=True)
    if plot :
        limit = np.max(np.abs(result.impact.values)) 
        limit = limit + 0.1*limit
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim([-limit,limit])
        ax.barh(result.index, result['impact'], color='blue')
        container = ax.containers[0]
        if show_pourcentage :
            ax.bar_label(container, labels=[f'{x:,.2f} %' for x in list(result.pourcentage_impact)])
        ax.plot([-1, 1], [impacting_variable_line, impacting_variable_line], "k--",color='green')
        if show_pourcentage :
            ax.set_xlabel("pourcentage of deviation")
        else :
            ax.set_xlabel("deviation")
        ax.set_ylabel("features")
    else :
        return result
