import torch
import torch.nn as nn

def kld_loss(recon_x, x, mu, logvar, temperature=1.0):
    BCE = nn.BCELoss()(recon_x, x)
    KLD = -temperature * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#The function returns the weighted sum of BCE and KLD losses.


def nll_loss(recon_x, x):
    return nn.NLLLoss()(recon_x, x)


def mse_loss_fn(recon_x, x):
    return nn.MSELoss()(recon_x, x)


def bce_loss_fn(recon_x, x):
    return nn.BCELoss()(recon_x, x)
