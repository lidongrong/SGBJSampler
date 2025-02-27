import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
This script implements our sampler and other competitive samplers. 
Though they are named with 'Optimizer', they are indeed samplers. Naming them optimizer is just due to convention.
"""


class SGLDOptimizer(torch.optim.Optimizer):
    """
    SGLD Sampler
    """
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
                #p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p,alpha = -group['lr'])

                if group['add_noise']:
                    noise = torch.randn_like(p.data) * torch.sqrt(torch.tensor(group['lr']))
                    p.data.add_(noise)
        return loss

class SGBOptimizer(torch.optim.Optimizer):
    """
    The proposed Sampler
    """
    def __init__(self, params, lr=1e-2, add_noise=True):
        defaults = dict(lr=lr, add_noise=add_noise)
        super(SGBOptimizer, self).__init__(params, defaults)

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
    """
    Stochastic Gradient Hamiltonian Monte Carlo Sampler 
    """

    def __init__(self, params, lr=1e-2, momentum=0.1, weight_decay=0, num_burn_in_steps=3000):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, num_burn_in_steps=num_burn_in_steps)
        super(SGHMCOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single SGHMC optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'])
                
                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['momentum'])
                
                if 'num_burn_in_steps' in group and group['num_burn_in_steps'] > 0:
                    # Burn-in updates
                    p.data.add_(-group['lr'] * d_p)
                    group['num_burn_in_steps'] -= 1
                else:
                    # Sampling steps
                    noise = torch.randn_like(d_p)
                    p.data.add_(buf, alpha=-group['lr']).add_(noise, alpha=torch.sqrt(torch.tensor(2.0 * group['lr'] * (1 - group['momentum']))))
                    
        return loss

class SGNHTOptimizer(torch.optim.Optimizer):
    """
    SGNHT Sampler
    """
    def __init__(self, params, lr=1e-2, A=0.01, B=0.1, add_noise=True):
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

