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

class SGLDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, add_noise=True):
        defaults = dict(lr=lr, add_noise=add_noise)
        super(SGLDOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

                if group['add_noise']:
                    noise = torch.randn_like(p.data) * torch.sqrt(2 * torch.tensor(group['lr']))
                    p.data.add_(noise)
        return loss

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


class SGHMCOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=.5, beta=.5, num_leapfrog_steps=1, add_noise=True):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if beta < 0.0:
            raise ValueError("Invalid beta value: {}".format(beta))
        if not isinstance(num_leapfrog_steps, int) or num_leapfrog_steps < 1:
            raise ValueError("Invalid number of leapfrog steps: {}".format(num_leapfrog_steps))

        defaults = dict(lr=lr, alpha=alpha, beta=beta, num_leapfrog_steps=num_leapfrog_steps, add_noise=add_noise)
        super(SGHMCOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if 'momentum' not in state:
                    state['step'] = 0
                    # Sample initial momentum from a normal distribution
                    state['momentum'] = torch.randn_like(p.data)

                momentum = state['momentum']

                # Perform leapfrog updates
                for _ in range(group['num_leapfrog_steps']):
                    d_p = p.grad.data

                    # Update momentum with gradient
                    momentum.add_(-group['lr'], d_p)

                    # Apply friction
                    momentum.mul_(1 - group['alpha'])

                    # Update parameter
                    p.data.add_(momentum)

                    # Re-evaluate loss and gradient if not last step
                    if _ < group['num_leapfrog_steps'] - 1 and closure is not None:
                        loss = closure()

                    if group['add_noise']:
                        # Inject noise to simulate the stochastic process
                        noise_std = torch.sqrt(torch.tensor(2 * group['alpha'] * group['beta'] * group['lr']))
                        noise = torch.randn_like(p.data) * noise_std
                        p.data.add_(noise)

                state['step'] += 1

        return loss

"""
class SGHMCOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.01, beta=0.1, num_leapfrog_steps=1, add_noise=True):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if beta < 0.0:
            raise ValueError("Invalid beta value: {}".format(beta))
        if not isinstance(num_leapfrog_steps, int) or num_leapfrog_steps < 1:
            raise ValueError("Invalid number of leapfrog steps: {}".format(num_leapfrog_steps))

        defaults = dict(lr=lr, alpha=alpha, beta=beta, num_leapfrog_steps=num_leapfrog_steps, add_noise=add_noise)
        super(SGHMCOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.randn_like(p.data)

                momentum = state['momentum']
                lr = group['lr']
                alpha = group['alpha']
                beta = group['beta']
                num_leapfrog_steps = group['num_leapfrog_steps']
                add_noise = group['add_noise']

                # Perform leapfrog updates
                for _ in range(num_leapfrog_steps):
                    d_p = p.grad.data

                    # Update momentum with gradient
                    momentum.add_(-lr * d_p)

                    # Apply friction
                    momentum.mul_(1 - alpha)

                    # Update parameter
                    p.data.add_(momentum)

                    # Re-evaluate loss and gradient if not last step
                    if _ < num_leapfrog_steps - 1 and closure is not None:
                        loss = closure()

                    if add_noise:
                        # Inject noise to simulate the stochastic process
                        noise_std = math.sqrt(2 * alpha * beta * lr)
                        noise = torch.randn_like(p.data) * noise_std
                        p.data.add_(noise)

                state['step'] += 1

        return loss
"""

class SGNHTOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, A=0.1, B=1, add_noise=True):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if A < 0.0:
            raise ValueError(f"Invalid A value: {A}")
        if B < 0.0:
            raise ValueError(f"Invalid B value: {B}")

        defaults = dict(lr=lr, A=A, B=B, add_noise=add_noise)
        super(SGNHTOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    state['xi'] = torch.zeros_like(p.data)

                state['step'] += 1
                momentum = state['momentum']
                xi = state['xi']
                grad = p.grad.data

                # Update xi, which is the Nosé-Hoover additional term (thermostat)
                xi.add_((torch.square(momentum).mean() / group['B'] - 1) * group['lr'])

                # Update momentum with gradient and friction
                momentum.add_(-group['lr'] * grad - xi * momentum * group['lr'])

                # Update the parameter
                p.data.add_(momentum * group['lr'])

                if group['add_noise']:
                    # Inject noise to simulate the stochastic process
                    noise_std = torch.sqrt(torch.tensor(2 * group['A'] * group['lr']))
                    noise = torch.randn_like(p.data) * noise_std
                    p.data.add_(noise)

        return loss

