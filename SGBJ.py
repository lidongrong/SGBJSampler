# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:49:25 2024

@author: lidon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

class SGBJOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, add_noise=True):
        defaults = dict(lr=lr, add_noise=add_noise)
        super(SGBJOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                #p.data.add_(-group['lr'], d_p)
                noise = torch.randn_like(p.data) * torch.sqrt(torch.tensor(group['lr']))
                d_p = p.grad.data
                #p.data.add_(noise)
                prod = noise * (d_p)
                prob = 0 - torch.logsumexp(torch.stack([torch.zeros_like(prod), prod]), dim=0)

                trial = torch.log(torch.rand(prob.size(), device=prob.device))
                direction = 2 * (trial <= prob) - 1

                update = direction * noise
                p.data.add_(update)

        return loss



