# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:03:07 2024

@author: lidon
"""

import torch
import torch.optim as optim
import torch.distributions as dist
from optimizers import SGBJOptimizer, SGHMCOptimizer, SGNHTOptimizer, SGLDOptimizer
from torch.distributions import Normal, MixtureSameFamily, Categorical
import numpy as np
import matplotlib.pyplot as plt

class BayesianModel:
    def __init__(self, initial_theta=0, learning_rate=0.01,optimizer = optim.Adam):
        self.theta = torch.tensor([initial_theta], requires_grad=True)
        self.optimizer = optimizer([self.theta], lr=learning_rate)
        self.theta_values = []  # To store theta values for tracing

    def log_likelihood(self, x):
        #return Normal(self.theta, 1).log_prob(x).sum() 
        return 0
    
    def log_prior(self):
        component_means = torch.tensor([-3, 3])
        component_scales = torch.tensor([1, 1])
        mixture_weights = torch.tensor([0.5, 0.5])
        
        mixture_distribution = MixtureSameFamily(
            Categorical(mixture_weights),
            Normal(component_means, component_scales)
        )
        
        return mixture_distribution.log_prob(self.theta)

    def neg_log_posterior(self, x):
        return -(self.log_likelihood(x) + self.log_prior())
    
    def log_posterior(self,x):
        return self.log_likelihood(x) + self.log_prior()

    def step(self, x):
        self.optimizer.zero_grad()
        loss = self.neg_log_posterior(x)
        loss.backward()
        #print('gradient: ', self.theta.grad)
        self.optimizer.step()

        # Check for NaN in the updated theta or in the neg_log_posterior
        if torch.isnan(self.theta).any() or torch.isnan(loss):
            # Revert to the last valid theta value
            self.theta.data = torch.tensor([self.theta_values[-1]], requires_grad=True)
        else:
            # Accept the new theta and record it
            self.theta_values.append(self.theta.item())
        return loss.item()

    def train(self, data, num_steps=2000):
        for step in range(num_steps):
            loss = self.step(data)
            if step % 100 == 0:
                print(f"Step {step}: theta = {self.theta.item():.4f}, Loss = {loss:.4f}")

    def plot_trace(self):
        plt.plot(self.theta_values)
        plt.xlabel('Step')
        plt.ylabel('Theta')
        plt.title('Trace of Theta during Optimization')
        plt.show()

    def plot_posterior_landscape(self, data, a, num_points=100,ax=None):
        # Generate theta values and compute the negative log posterior
        thetas = np.linspace(-a, a, num_points)
        log_posteriors = []
        for theta in thetas:
            self.theta = torch.tensor([theta], requires_grad=False)
            log_posterior = self.neg_log_posterior(data).item()
            log_posteriors.append(-log_posterior)
    
        # If no ax is provided, create a new figure and axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            created_figure = True
        else:
            created_figure = False
        
        # Plotting the log posterior on the main axis
        ax.plot(thetas, log_posteriors, label='Log Posterior', zorder=1)
        
        # Plotting the optimization trace on the main axis
        trace_evaluations = []
        for theta in self.theta_values:
            self.theta = torch.tensor([theta], requires_grad=False)
            trace_evaluation = -self.neg_log_posterior(data).item()
            trace_evaluations.append(trace_evaluation)
        
        ax.scatter(self.theta_values, trace_evaluations, color='red', marker='x', s=10, zorder=2, label='Optimization Trace')
        
        # Setting main axis properties
        ax.set_xlim(-a, a)
        ax.set_ylim(min(log_posteriors), 0)
        ax.set_title(f'Optimizer: {self.optimizer.__class__.__name__}, Step Size: {self.optimizer.defaults["lr"]}')
        ax.grid(True)
        ax.legend()
    
        # Create an inset axis for the zoomed view
        ax_inset = ax.inset_axes([0.05, 0.05, 0.3, 0.3])  # [left, bottom, width, height] relative to the main axis
        ax_inset.plot(thetas, log_posteriors, zorder=1)
        ax_inset.scatter(self.theta_values, trace_evaluations, color='red', marker='x', s=50, zorder=2)
        ax_inset.set_title('Zoomed View')
        ax_inset.grid(True)
    
        # Only show the plot if no external ax is provided
        if created_figure:
            plt.show()


# True parameter for data generation
true_theta = torch.tensor(0.)

# Generate some data
#torch.manual_seed(42)
data = Normal(true_theta, 1).sample(sample_shape=(1,))

#step_sizes = [1,0.5,0.1,0.01,0.001,0.0001,0.00001,0.000001]
step_sizes = [1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005]
optimizers = [SGBJOptimizer,SGLDOptimizer,SGHMCOptimizer,SGNHTOptimizer]
optimizer_names = ['SGBJ','SGLD','SGHMC','SGNHT']

n_rows = len(step_sizes)
n_cols = len(optimizers)
for step_size in step_sizes:
    # Create a new figure for each step size
    fig, axes = plt.subplots(1, len(optimizers), figsize=(5 * len(optimizers), 5))
    
    for i, optimizer in enumerate(optimizers):
        # Create model with current optimizer and step size
        model = BayesianModel(initial_theta=np.random.normal(), #2., 
                              learning_rate=step_size, optimizer=optimizer)
        model.train(data,num_steps = 8000)
        
        # Get current axis
        if len(optimizers) > 1:
            ax = axes[i]  # If multiple optimizers, use subplot axes
        else:
            ax = axes      # If one optimizer, axes is not an array
        
        # Plot on the designated axis
        model.plot_posterior_landscape(data, a=5, ax=ax)
        
        # Set specific titles or labels if needed
        ax.set_title(f'{optimizer_names[i]}')

    # Adjust layout and display the figure
    plt.savefig(f'images/Multimodal/stepsize{step_size}.png')
    plt.show()
    

