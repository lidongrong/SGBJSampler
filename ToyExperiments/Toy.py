import torch
import torch.distributions as dist
from optimizers import SGBJOptimizer, SGHMCOptimizer, SGNHTOptimizer, SGLDOptimizer
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
#torch.manual_seed(3407)

class BayesianNormalModel2D:
    def __init__(self, prior_mu, prior_sigma, likelihood_cov):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.likelihood_cov = torch.diag_embed(likelihood_sigma)
        self.posterior_mu = torch.tensor(prior_mu, requires_grad=True)

    def register_data(self, data):
        self.data = data
        # Calculate sample statistics
        self.sample_mean = torch.mean(data, dim=0)
        self.n_samples = data.shape[0]
        
        # Calculate posterior parameters analytically
        # Posterior precision = prior precision + n * likelihood precision
        prior_precision = 1 / (self.prior_sigma ** 2)
        likelihood_precision = 1 / torch.diagonal(self.likelihood_cov)
        self.posterior_precision = prior_precision + self.n_samples * likelihood_precision
        self.posterior_std = torch.sqrt(1 / self.posterior_precision)
        
        # Posterior mean = (prior_precision * prior_mean + n * likelihood_precision * sample_mean) / posterior_precision
        self.analytical_posterior_mean = (
            prior_precision * self.prior_mu + 
            self.n_samples * likelihood_precision * self.sample_mean
        ) / self.posterior_precision

    def log_posterior(self, data):
        """
        Calculate log posterior directly using analytical form
        The posterior is also a Normal distribution
        """
        posterior_dist = dist.MultivariateNormal(
            self.analytical_posterior_mean,
            torch.diag(self.posterior_std ** 2)
        )
        return posterior_dist.log_prob(self.posterior_mu)
    
    def log_density(self, x, y):
        self.prior_mu[0] = x
        self.prior_mu[1] = y
        return self.log_posterior(self.data)
    
    def meshgrid_log_density(self, x, y):
        """
        Evaluate log density at points specified by x and y
        
        Parameters:
        x, y: numpy arrays or torch tensors (including meshgrids)
        
        Returns:
        numpy array of log densities
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Initialize results
        results = torch.zeros_like(x_flat)
        
        # Store original prior_mu
        original_prior_mu = self.prior_mu.clone()
        
        # Evaluate for each point
        for i in range(len(x_flat)):
            self.prior_mu[0] = x_flat[i]
            self.prior_mu[1] = y_flat[i]
            results[i] = self.log_posterior(self.data)
        
        # Restore original prior_mu
        self.prior_mu = original_prior_mu
        
        # Reshape results to match input shape
        results = results.reshape(original_shape)

        results = results.detach().numpy()
        
        return results
    

# Function to generate synthetic data
def generate_synthetic_data(true_mu, likelihood_sigma, num_samples=10000):
    return dist.MultivariateNormal(true_mu, torch.diag(likelihood_sigma) ** 2).sample((num_samples,))

# Function to perform Bayesian inference
def bayesian_inference(model, data, optimizer, num_iterations=50):
    model.register_data(data)
    samples = []

    samples = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        negative_log_posterior = -model.log_posterior(data)
        negative_log_posterior.backward()
        optimizer.step()

        # Retrieve the current sample
        current_sample = model.posterior_mu.detach()

        # Check for NaN or Inf
        if torch.isnan(current_sample).any() or torch.isinf(current_sample).any():
            print("NaN or Inf encountered in sampling at iteration", iteration)
            break  # Exit the sampling loop

        samples.append(current_sample.numpy().copy())
        if (iteration+1) % 1 == 0:
            print(f"Iteration {iteration+1}: µ={current_sample}")

    #print('samples: ', samples)
    return samples

# Hyperparameters and model setup
true_mu = torch.tensor([0.5, -0.5])
likelihood_sigma = torch.tensor([16., 16.])  # Assuming we know the true sigma for both dimensions
prior_mu = torch.tensor([0.5, -0.5])
prior_sigma = torch.tensor([10.,10.])
learning_rate = 0.001

# Generate synthetic data
data = generate_synthetic_data(true_mu, likelihood_sigma,num_samples=750)

# Initialize the model
model = BayesianNormalModel2D(prior_mu, prior_sigma, likelihood_sigma)

optimizer_classes = [SGBJOptimizer, SGLDOptimizer, SGHMCOptimizer, SGNHTOptimizer]
step_sizes = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4]
num_iterations = 500
results = {}

for optimizer_class in optimizer_classes:
        for step_size in step_sizes:
            model = BayesianNormalModel2D(prior_mu, prior_sigma, likelihood_sigma)
            optimizer =  optimizer_class([model.posterior_mu], lr=step_size)
            tmp = np.array([0.,0.])
            bayesian_estimate_mu = bayesian_inference(
                model, data, optimizer, num_iterations
            )
            bayesian_estimate_mu = [tmp] + bayesian_estimate_mu
            # Save the results
            results[(optimizer_class.__name__, step_size)] = bayesian_estimate_mu
            print(f"Optimizer: {optimizer_class.__name__}, Step Size: {step_size}, "
                  f"Estimate: {bayesian_estimate_mu}")

def compute_density(x, y):
    # Your actual density computation would go here
    return np.exp(-((x-0.5)**2 + (y+0.5)**2))


# analyze the results
# Define the optimizer names
optimizer_names = ['SGBJOptimizer', 'SGLDOptimizer', 'SGHMCOptimizer', 'SGNHTOptimizer']
display_names = ['SGBJ Sampler', 'SGLD Sampler', 'SGHMC Sampler','SGNHT Sampler']

# Function to plot the sampling paths in separate figures for each step size
# Function to plot the sampling paths with density contours
def plot_sampling_paths_with_density(results, optimizer_names, true_mode, display_names,model = model,num_contours=50):
    # Get unique step sizes
    step_sizes = sorted(set(key[1] for key in results.keys()))

    for step_size in step_sizes:
        # Create a new figure for each step size
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Sampling Paths for Step Size: {step_size}', fontsize=16)

        # Create a grid of x and y values to calculate the density for the contour
        x_min, x_max = true_mode[0] - 1, true_mode[0] + 1
        y_min, y_max = true_mode[1] - 1, true_mode[1] + 1
        x = np.linspace(x_min, x_max, 200)
        y = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x, y)
        """
        model.register_data(data)
        X_torch = torch.tensor(X, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32)
        Z_torch = X_torch

        # Evaluate the function using PyTorch
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                print(f'rendering {i}{j} th plot...')
                Z_torch[i, j] = model.log_density(X[i, j], Y[i, j])
        """
        Z= compute_density(X,Y)
        #print('evaluating the meshgrid...')
        #Z = model.meshgrid_log_density(X, Y)
        k=0

        for j, optimizer_name in enumerate(optimizer_names):
            key = (optimizer_name, step_size)
            if key in results:
                samples = np.array(results[key])
                ax = axes[j]

                # Plot the density contour
                ax.contour(X, Y, Z, levels=num_contours, cmap='viridis')

                # Plot the sampling path
                ax.plot(samples[:, 0], samples[:, 1], marker='o', markersize=3, label=display_names[k])
                ax.set_title(f'{display_names[k]}')
                ax.set_xlabel('Parameter 1')
                ax.set_ylabel('Parameter 2')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.grid(True)
                ax.legend()

                # Determine the original scale from the samples
                full_x_min, full_x_max = samples[:, 0].min(), samples[:, 0].max()
                full_y_min, full_y_max = samples[:, 1].min(), samples[:, 1].max()

                # Create an inset axis with the original scale
                ax_inset = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
                ax_inset.plot(samples[:, 0], samples[:, 1], marker='o', markersize=1)
                max_x = max(abs(full_x_min), abs(full_x_max))
                max_y = max(abs(full_y_min), abs(full_y_max))
                ax_inset.set_xlim( -2 * max_x, 2 * max_x)
                ax_inset.set_ylim(-2 * max_y, 2 * max_y)
                ax_inset.grid(True)

                # Optionally, you can add a border around the inset
                # to make it more distinct
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(1)
            k=k+1

        # Adjust the layout and display the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make space for suptitle
        plt.show(block=True)
        #plt.savefig(f'images/Step{step_size:.0e}.jpg')

# Assuming true_mode is known and specified
true_posterior_mode = (0.5, -0.5)  # Replace with the actual mode coordinates

# Run the function to plot the sampling paths with density contours
plot_sampling_paths_with_density(results, optimizer_names, true_posterior_mode,display_names)