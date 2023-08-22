# Number crunching
import numpy as np
import scipy.stats as stats

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# GP stuff
import gpflow as gp
from gpflow.utilities import print_summary
from gpflow.likelihoods import MultiLatentTFPConditional
from typing import Callable, Optional, Type
from gpflow.utilities import positive

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.gridspec as gridspec

# Misc
import sys
import time

from Data import x, y, xx, yy

# Squeeze y and yy a tiny bit (linearly) as Beta breaks at absolute 0 and 1
y = y * (1 - 1e-4) + 0.5e-4
yy = yy * (1 - 1e-4) + 0.5e-4
# ---------------------------------------------------------------------------------------------
# Beta likelihood
# ---------------------------------------------------------------------------------------------

# define likelihood function for GPflow
class BetaTFPConditional(MultiLatentTFPConditional):

    def __init__(
            self,
            distribution_class: Type[tfp.distributions.Distribution] = tfp.distributions.Beta,
            scale_transform: Optional[tfp.bijectors.Bijector] = None,
            **kwargs,
    ):
        if scale_transform is None:
            scale_transform = positive(base="exp")
        self.scale_transform = scale_transform

        def conditional_distribution(Fs) -> tfp.distributions.Distribution:
            tf.debugging.assert_equal(tf.shape(Fs)[-1], 2)
            alpha = self.scale_transform(Fs[..., :1])
            beta = self.scale_transform(Fs[..., 1:])
            return distribution_class(alpha, beta)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )

# ---------------------------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------------------------

likelihood = BetaTFPConditional()

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

P = 2
L = 2
kernel = gp.kernels.LinearCoregionalization(
    [
        gp.kernels.Matern52(lengthscales=0.1) + gp.kernels.Linear(),  # This is k1, the kernel of f1
        gp.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ], W=np.random.randn(P, L)
)

M = 10  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z1 = np.linspace(x.min(), x.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gp.inducing_variables.SharedIndependentInducingVariables(
    gp.inducing_variables.InducingPoints(Z1),  # Sharing inducing points
)

model = gp.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
)

# ---------------------------------------------------------------------------------------------
# Minibatch and optimisation
# ---------------------------------------------------------------------------------------------

minibatch_size = 500

x_new = x[~np.isnan(x).any(axis=1)]
y_new = y[~np.isnan(y).any(axis=1)]

# Squeezes y a tiny bit (linearly) as Beta breaks at absolute 0 and 1
data = (x_new, y_new)
train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(minibatch_size * 50)

train_iter = iter(train_dataset.batch(minibatch_size))

# We turn off training for inducing point locations
gp.set_trainable(model.inducing_variable, False)

training_loss = model.training_loss_closure(train_iter, compile=True)
optimizer = tf.optimizers.Adam(0.1)


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)

epochs = 500

for step in range(1, epochs + 1):
    optimization_step()
    print(f"Epoch {step} - Loss: {-training_loss().numpy() : .4f}")

# ---------------------------------------------------------------------------------------------
# Extracting distributions from sampled posterior
# ---------------------------------------------------------------------------------------------

# define number of samples taken from the posterior
nsamp = 500

# Pull samples from posterior
Fss = model.predict_f_samples(xx, nsamp, full_cov=False, full_output_cov=True).numpy()

# Define transform used to create alpha and beta
trans = gp.utilities.bijectors.positive(base="exp")

# preallocate empty arrays to make life easier and quicker
alpha = np.zeros((len(xx), nsamp))
beta = np.zeros((len(xx), nsamp))
mean = np.zeros((len(xx), nsamp))

# Number of samples per distribution at each
beta_samples = 100

# preallocate empty array to make life easier and quicker
dist_old = np.empty((beta_samples, len(xx), nsamp))

print("Calculating the Beta dist, mean and var per test point, this can take some time..")
alpha = trans(tf.squeeze(Fss[:, :, 0]))
beta = trans(tf.squeeze(Fss[:, :, 1]))
dist = np.array(tfd.Beta(alpha,beta).sample(beta_samples)) # create beta dist per posterior sample and sample from it

total_mean = np.mean((alpha / (alpha + beta)), axis=0) # total mean across no of posterior samles
total_var = np.mean(np.var(dist, axis=0), axis=0) + np.var(np.mean(dist, axis=0), axis=0) # total variance

# final_alpha and beta
final_alpha = total_mean ** 2 * ((1 - total_mean) / total_var - 1 / total_mean)
final_beta = final_alpha * (1 / total_mean - 1)

# stick back into Beta distribution
final_dist = tfd.Beta(final_alpha, final_beta)
print("Done!")
# final_dist = tfd.Beta(final_alpha, final_beta).sample(100)

# ---------------------------------------------------------------------------------------------
# Judging the model
# ---------------------------------------------------------------------------------------------

# log likelihood calculation, this is currently not working for the given data
log_likelihood_yy = np.sum(final_dist.log_prob(yy)) 
print('Sum of the log likelihoods:', log_likelihood_yy)

# calculate the error and turn it into a matrix so multiplication works
y_yM_error = np.matrix(yy-total_mean.reshape(len(total_mean), 1))

NMSE = np.array((100/(len(xx)*np.var(yy)))*np.sqrt(y_yM_error.T*y_yM_error))
print(NMSE)

# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------

plt.figure()
line1 = plt.plot(xx, np.array(final_dist.sample(beta_samples)).T, ".", mew=.005, color='gray', alpha=0.02)
line2 = plt.plot(xx, yy, ".", color='deepskyblue', markersize=0.8, label="Training points")
line3 = plt.plot(xx, total_mean, color='k', lw=1, alpha=.9, label="Prediction mean")

plt.xlabel('Normalised Wind Speed')
plt.ylabel('Normalised Power')

plt.axhline(y=1, color='k', ls='--')

legend_elements = [Line2D([0], [0], color="white", marker='.', markerfacecolor="gray", markersize=10,
                          label="Beta dist sample \nat test point"),
                   Line2D([0], [0], color="white", marker='.', markersize=10, markerfacecolor="deepskyblue",
                          label="Test data"),
                   Line2D([0], [0], color='black', lw=2, label='Mean of prediction')]

plt.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(-0.5,1.4,1,0.4),mode='expand',frameon=False)

plt.show()
