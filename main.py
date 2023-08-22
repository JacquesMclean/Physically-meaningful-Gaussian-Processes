import numpy as np
import matplotlib.pyplot as plt
import sys

# define Density, swept area of turbine, velocity vector, rated power and performance factor 
# Data size is 17k as this is dataset size in paper
# Add noise to density to badly simulate noisy power curve
d = np.full((1000),1.3) + np.random.uniform(low=-0.2, high=0.2, size=1000)
A = 80 ** 2 * np.pi
V = np.linspace(0, 15, 1000)
Cp = 0.4
rated_pow = 3e6 # both Cp and rated_power are arbitrary 

# power output equation and cap power to rated power
P = 0.5*Cp*d*A*V**3
P = np.where(P < rated_pow, P, rated_pow)

# normalise input between 0 and 1
def normalise(vector):
    normed = ((vector - np.min(vector)) / (np.max(vector) - np.min(vector))).reshape(len(vector), 1)
    return normed

# following standard notation: x and y are training, xx and yy are test
# the astute amoungst you will notice that I am predicting the training data. This is not the case in the paper but because this is a demonstration synthetic data set there is no need
x = normalise(V)
y = normalise(P)
xx = normalise(V)
yy = normalise(P)

