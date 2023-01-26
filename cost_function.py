# created by Jonathan Jeremie Randriarison at 20230206 22:13.
# 
# Multi (Variationnal)Autoencoder for fault detection and isolation
# Fault explainability

# define cost function 
def criterion(elem, recon, model , model_type="AE", bias_in_wrong_class=None, mode='test'):
    # L2 regularization coefficient 
    l2_lambda = 0.001
    l2_norm = sum(p.pow(2.0).sum()
                  for p in model.parameters())
    L2_regularization = l2_lambda*l2_norm
    if model_type == "VAE" :
        if mode == 'test':
            loss = ((elem - recon)**2).sum() + model.kl
        elif mode == 'train' :
            recon  = recon + bias_in_wrong_class[:,None]
            loss = ((elem - recon)**2).sum() + model.kl + L2_regularization
    elif model_type == "AE" :
        if mode == 'test' :
            loss = ((elem - recon)**2).sum()
        elif mode == 'train' :
            l2_lambda = 0.01
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            recon  = recon + bias_in_wrong_class[:,None] 
            loss = ((elem - recon)**2 ).sum() + L2_regularization
    return loss