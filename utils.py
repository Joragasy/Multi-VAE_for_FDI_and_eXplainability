# created by Jonathan Jeremie Randriarison at 20230206 22:12.
# 
# Multi (Variationnal)Autoencoder for fault detection and isolation
# Fault explainability

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , f1_score
from cost_function import criterion
from dataset import create_dataset

# we use this function for plotting confusion matrix
def plot_confusion_matrix(y_pred, y_true) :
    cf_matrix = confusion_matrix( y_pred , y_true)
    # plt.figure(figsize = (15,7))
    fig = sns.heatmap(cf_matrix , annot=True ,fmt = ".0f", cmap='Blues' , annot_kws={"size": 10})
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.title("Confusion matrix") 
    plt.show(fig)

# model trainer : create training phase routine for all instance of autoencoder or variational autoencoder
def train(train_data_loader,Model,model_type,model_num,num_epochs = 30 , device="cpu"):
    optimizer = torch.optim.Adam(Model.parameters() , lr=1e-3 , weight_decay=1e-5 )
    outputs = []
    show_elem = None
    for epoch in range(num_epochs) :
        for elem , target in train_data_loader :
            elem = elem.to(device)
            recon = Model(elem)[0]
            bias_in_wrong_class = torch.where(target==model_num,0,3.9).to(device)
            loss = criterion(recon,elem,Model,model_type=model_type,bias_in_wrong_class=bias_in_wrong_class,mode='train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 :
            print(f"epoch : {epoch +1} , loss : {loss.item():4f}")
        outputs.append((epoch, elem, recon))
    return Model

# this function help us to preprocess and predict from pandas Series (X) , nb : used  pandas "apply" or dataframe.iloc as input (X)
def inference(X, model , detect_new_class=False):
    # convert pandas Series to tensor ( Pytorch )
    data_loader = create_dataset(X.to_frame().T,batch_size=1)
    for in_data , _ in data_loader :
        Input = in_data
    res = model.predict(Input,threshold=10,detect_new_class=detect_new_class)
    if  detect_new_class :
        return res
    else :
        res = res[1]+ [res[0]]
        prediction_with_recon_loss = pd.Series(res)
        #prediction_with_recon_loss = res
        return prediction_with_recon_loss