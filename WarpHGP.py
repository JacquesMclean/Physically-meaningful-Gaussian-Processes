# Number crunching
import numpy as np
import scipy.stats as stats

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

# GP stuff
import gpflow as gp

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Misc
import sys

from main import x, y, xx, yy

# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def warper(input):
    input = np.where(input==1,0.99,input)
    input = np.where(input==0,0.01,input)
    return np.log(input[:, 0] / (1 - input[:, 0]))

# warp train and test
y_warp = warper(y).reshape(len(y),1)
yy_warp = warper(yy).reshape(len(yy),1)

# ---------------------------------------------------------------------------------------------
# GP stuff
# ---------------------------------------------------------------------------------------------

likelihood = gp.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

kernel = gp.kernels.SeparateIndependent(
    [
        gp.kernels.SquaredExponential(lengthscales=0.01),  # This is k1, the kernel of f1
        gp.kernels.Matern32(),  # this is k2, the kernel of f2
    ]
)

M = 10  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z = np.linspace(x.min(), x.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gp.inducing_variables.SeparateIndependentInducingVariables(
    [
        gp.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
        gp.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
    ]
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

# define size of minibatch
minibatch_size = 50

# put x and y into data array, add jitter to help GPflow not have a heart attack
x_new = x[~np.isnan(x).any(axis=1)]
y_new = y_warp[~np.isnan(y_warp).any(axis=1)]
data = (x_new, y_new)

# set seed
tf.random.set_seed(1)

# divide into datasets
train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(minibatch_size * 500)

train_iter = iter(train_dataset.batch(minibatch_size))

# We turn off training for inducing point locations
gp.set_trainable(model.inducing_variable, False)

training_loss = model.training_loss_closure(train_iter, compile=True)
optimizer = tf.optimizers.Adam(0.1)


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)

epochs = 1500

for step in range(1, epochs + 1):
    optimization_step()
    print(f"Epoch {step} - Loss: {-training_loss().numpy() : .4f}")

# ---------------------------------------------------------------------------------------------
# Pull results and unwarp
# ---------------------------------------------------------------------------------------------

# Pull results
Ymean, Yvar = model.predict_y(xx)
Ymean = Ymean.numpy().squeeze()
Ystd = tf.sqrt(Yvar).numpy().squeeze()

# Unwarp mean and test data
Final_mean = np.exp(Ymean) / (1 + np.exp(Ymean))

# Create Mx/min lines in order to show uncertainty
maxL_warp = Ymean + 2 * Ystd
minL_Warp = Ymean - 2 * Ystd

# Unwarp max and min lines
maxL = np.exp(maxL_warp) / (1 + np.exp(maxL_warp))
minL = np.exp(minL_Warp) / (1 + np.exp(minL_Warp))

# ---------------------------------------------------------------------------------------------
# Judging the model, this is all done in  warp space
# ---------------------------------------------------------------------------------------------

# reshape for numpy
Yvar_2D = np.array(Yvar).reshape(len(Yvar),1)
Ymean_2D = np.array(Ymean).reshape(len(Ymean),1)
Final_mean_2D = np.array(Final_mean).reshape(len(Final_mean),1)

# breaking up the long calculation into 3 parts
part1 = -0.5*np.log(Yvar_2D*2*np.pi)
part2 = (yy_warp - Ymean_2D)**2
part3 = 2*Yvar_2D

# Calculate and sum the likelihood
log_likelihood_sum = np.sum(np.array(part1 - part2/part3))

print('Sum of the log likelihoods:', log_likelihood_sum)

# NMSE
y_yM_error = np.matrix(yy_warp-Ymean_2D)

NMSE = np.array((100/(len(xx)*np.var(yy)))*np.sqrt(y_yM_error.T*y_yM_error))
print('NMSE:',NMSE)

# ---------------------------------------------------------------------------------------------
# Warped figure
# ---------------------------------------------------------------------------------------------

plt.figure()
plt.plot(xx,yy_warp, ".", markersize=0.2, color='k')
plt.plot(xx,Ymean, color='green', lw=1, alpha=.9)

plt.fill_between(
    xx[:, 0],
    maxL_warp,
    minL_Warp,
    color="lightgreen",
    alpha=0.2,
)

plt.xlabel('Normalised Wind Speed')
plt.ylabel('Warped Power')

legend_elements1 = [Line2D([0], [0], color="white", marker='.', markerfacecolor="k", markersize=10,
                          label="Test data"),
                   Line2D([0], [0], color='forestgreen', lw=1, label='Mean of prediction'),
                   Patch(facecolor='limegreen', edgecolor='limegreen', alpha=0.2,
                         label='Uncertainty +- 2σ')]

plt.legend(handles=legend_elements1, loc="upper center", bbox_to_anchor=(-0.15,1.03,1,0.3),mode='expand',frameon=False)

# ---------------------------------------------------------------------------------------------
# Big picture HGP
# ---------------------------------------------------------------------------------------------

plt.figure()
plt.plot(xx,yy, ".", markersize=0.2, color='k')
plt.plot(xx,Final_mean, color='green', lw=1, alpha=.9)

plt.fill_between(
    xx[:, 0],
    maxL,
    minL,
    color="lightgreen",
    alpha=0.2,
)

plt.xlabel('Normalised Wind Speed')
plt.ylabel('Normalised Power')

plt.axhline(y=1, color='k', ls='--')

legend_elements2 = [Line2D([0], [0], color="white", marker='.', markerfacecolor="k", markersize=10,
                          label="Test data"),
                   Line2D([0], [0], color='forestgreen', lw=1, label='Mean of prediction'),
                   Patch(facecolor='limegreen', edgecolor='limegreen', alpha=0.2,
                         label='Uncertainty +- 2σ')]

plt.legend(handles=legend_elements2, loc="upper center", bbox_to_anchor=(-0.15,1.03,1,0.3),mode='expand',frameon=False)

plt.show()