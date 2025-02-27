import torch
import torch.distributions as dist
from optimizers import SGBJOptimizer, SGHMCOptimizer, SGNHTOptimizer, SGLDOptimizer
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import studentT


# Set random seed for reproducibility
torch.manual_seed(3407)

# Define the Bayesian model with Normal priors for the mean vector
class BayesianNormalModel2D:
    def __init__(self, prior_mu, prior_sigma, likelihood_cov):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.likelihood_cov = torch.diag_embed(likelihood_sigma)
        self.posterior_mu = torch.tensor(prior_mu, requires_grad=True)  # Parameter to learn

    def prior(self):
        T = studentT.StudentT
        return T(df = torch.tensor([2.0]),loc = self.prior_mu)

    def likelihood(self, data):
        T = studentT.StudentT
        return T(df = torch.tensor([2.0]),loc = self.posterior_mu)

    def log_posterior(self, data):
        # Log prior probability
        log_prior = self.prior().log_prob(self.posterior_mu).sum()
        # Log likelihood probability
        log_likelihood = self.likelihood(data).log_prob(data).sum()
        # Log posterior is the sum of the log prior and log likelihood
        return log_prior + log_likelihood
    
        

# Function to generate synthetic data
def generate_synthetic_data(true_mu, likelihood_sigma, num_samples=1000):
    T = studentT.StudentT
    return T(df = torch.tensor([2.0]),loc =true_mu).sample((num_samples,)) 

# Function to perform Bayesian inference
def bayesian_inference(model, data, optimizer, num_iterations=50):
    samples = []

    samples = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        negative_log_posterior = -model.log_posterior(data)
        print('negative log post:', negative_log_posterior)
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
likelihood_sigma = torch.tensor([1.0, 1.0])  # Assuming we know the true sigma for both dimensions
prior_mu = torch.tensor([0.0, 0.0])
prior_sigma = torch.tensor([1.0, 1.0])
learning_rate = 0.001

# Generate synthetic data
data = generate_synthetic_data(true_mu, likelihood_sigma)

# Initialize the model
model = BayesianNormalModel2D(prior_mu, prior_sigma, likelihood_sigma)

optimizer_classes = [SGBJOptimizer, SGLDOptimizer, SGHMCOptimizer, SGNHTOptimizer]
step_sizes = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4]
num_iterations = 50
results = {}

for optimizer_class in optimizer_classes:
        for step_size in step_sizes:
            model = BayesianNormalModel2D(prior_mu, prior_sigma, likelihood_sigma)
            optimizer = optimizer_class([model.posterior_mu], lr=step_size)
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
display_names = ['SGBJ', 'SGLD', 'SGHMC','SGNHT']

# Function to plot the sampling paths in separate figures for each step size
# Function to plot the sampling paths with density contours
def plot_sampling_paths_with_density(results, optimizer_names, true_mode, display_names,num_contours=50):
    # Get unique step sizes
    step_sizes = sorted(set(key[1] for key in results.keys()))

    for step_size in step_sizes:
        # Create a new figure for each step size
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Sampling Paths for Step Size: {step_size}', fontsize=16)

        # Create a grid of x and y values to calculate the density for the contour
        x_min, x_max = true_mode[0] - 1, true_mode[0] + 1
        y_min, y_max = true_mode[1] - 1, true_mode[1] + 1
        x = np.linspace(x_min, x_max, 400)
        y = np.linspace(y_min, y_max, 400)
        X, Y = np.meshgrid(x, y)
            
        Z = compute_density(X, Y)
        k=0

        for j, optimizer_name in enumerate(optimizer_names):
            key = (optimizer_name, step_size)
            if key in results:
                samples = np.array(results[key])
                ax = axes[j]

                # Plot the density contour
                ax.contour(X, Y, Z, levels=num_contours, cmap='viridis')

                # Plot the sampling path
                ax.plot(samples[:, 0], samples[:, 1], marker='o', markersize=3, label=optimizer_name)
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
        plt.show()
        plt.savefig(f'images/Ttoy{step_size}.png')

# Assuming true_mode is known and specified
true_posterior_mode = (0.5, -0.5)  # Replace with the actual mode coordinates

# Run the function to plot the sampling paths with density contours
plot_sampling_paths_with_density(results, optimizer_names, true_posterior_mode,display_names)