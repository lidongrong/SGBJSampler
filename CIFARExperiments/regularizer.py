import torch

def l2_regularization(model, lambda_reg,device):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)**2
    return (lambda_reg/2) * l2_reg