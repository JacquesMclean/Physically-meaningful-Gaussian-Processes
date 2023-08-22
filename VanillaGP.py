import numpy as np

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# GP stuff
import gpflow as gp

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Import data from main file
from main import x, y, xx, yy

# ---------------------------------------------------------------------------------------------
# GP stuff
# ---------------------------------------------------------------------------------------------

# Kernel
k1 = gp.kernels.Matern32()
k2 = gp.kernels.Linear()
k = k1 + k2

# Number of inducing locations
M = 250

 # Initialize inducing locations to the first M inputs in the dataset
Z = np.linspace(x.min(), x.max(), M)[:, None]

# define model
model = gp.models.SVGP(k, gp.likelihoods.Gaussian(), Z)

# ---------------------------------------------------------------------------------------------
# Minibatch and optimisation
# ---------------------------------------------------------------------------------------------

minibatch_size = 800

x_new = x[~np.isnan(x).any(axis=1)]
y_new = y[~np.isnan(y).any(axis=1)]
data = (x_new, y_new * (1 - 1e-4) + 0.5e-4)

# fix the random seed - using tensor and numpy
tf.random.set_seed(1)
train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(minibatch_size * 50)

train_iter = iter(train_dataset.batch(minibatch_size))

# We turn off training for inducing point locations
gp.set_trainable(model.inducing_variable, False)

# Set up compiler and optimiser
training_loss = model.training_loss_closure(train_iter, compile=True)
optimizer = tf.optimizers.Adam(0.1)

# define optimisation step
@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)


# define epochs/number of training iterations
epochs = 2500

for step in range(1, epochs + 1):
    optimization_step()
    print(f"Epoch {step} - Loss: {-training_loss().numpy() : .4f}")

# ---------------------------------------------------------------------------------------------
# Pull mean and variance
# ---------------------------------------------------------------------------------------------

# use predict y instead of predict f
Ymean, Yvar = model.predict_y(xx)

Ymean_2D = np.asarray(Ymean[:,0]).reshape(len(Ymean), 1)
Yvar_2D = np.asarray(Yvar[:,0]).reshape(len(Yvar), 1)

Ymean = np.asarray(Ymean[:,0])
Yvar = np.asarray(Yvar[:,0])

Ystd = np.asarray(np.sqrt(Yvar))

# ---------------------------------------------------------------------------------------------
# Judging the model
# ---------------------------------------------------------------------------------------------

# breaking up the long calculation into 3 parts because I have to reshape everything
part1 = -0.5*np.log(Yvar_2D*2*np.pi)
part2 = (yy - Ymean_2D)**2
part3 = 2*Yvar_2D

# Put it all together
hand_log_likelihood = np.array(part1 - part2/part3)

# Sum the likelihood
log_likelihood_sum = np.sum(hand_log_likelihood)

print('The result that matters and it better be small:', log_likelihood_sum)


# need to turn this into matrix so dot product works
y_yM_error = np.matrix(yy-Ymean_2D)

NMSE1 = np.array((100/(len(xx)*np.var(yy)))*(y_yM_error.T*y_yM_error))
NMSE2 = np.array((100/(len(xx)*np.var(yy)))*np.sqrt(y_yM_error.T*y_yM_error))

print(NMSE1)
print(NMSE2)

# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------

plt.figure(figsize=(6,6))
plt.plot(xx, yy, ".", markersize=0.2, color='k')
plt.plot(xx, Ymean, "darkred", lw=1)
plt.xlabel('Wind speed')
plt.ylabel('Normalised power')

# 1 Sigma fill
plt.fill_between(
    xx[:, 0],
    Ymean - 3 * Ystd,
    Ymean + 3 * Ystd,
    color="indianred",
    alpha=0.2,
)

plt.axhline(y=1, color='k', ls='--')  # nice horizontal line to make it easier to visualise rated power performance

# legend stuff
legend_elements = [Line2D([0], [0], color="white", marker='.', markerfacecolor="k", markersize=10,
                          label="Test data"),
                   Line2D([0], [0], color='darkred', lw=1, label='Mean of prediction'),
                   Patch(facecolor='indianred', edgecolor='indianred', alpha=0.2,
                         label='Uncertainty +- 2σ')]

# Create the legend
plt.legend(handles=legend_elements, loc='lower right')

plt.figure()
plt.plot(xx, hand_log_likelihood, ".", markersize=0.4, color='k')
plt.show()