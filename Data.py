import numpy as np
import matplotlib.pyplot as plt
import sys

data_size = 2000 # reccomended to keep low as Beta post processing is highly intensive

# define Density, swept area of turbine, velocity vector, rated power and performance factor 
# Add noise to density to badly simulate noisy power curve

def PC_gen(size):
    d = np.full((size),1.3) + np.random.uniform(low=-0.15, high=0.15, size=size)
    A = 80 ** 2 * np.pi
    V = np.linspace(0, 15, size)
    Cp = 0.4
    rated_pow = 3e6 # both Cp and rated_power are arbitrary 
    P = 0.5*Cp*d*A*V**3
    return normalise(V), normalise(np.where(P < rated_pow, P, rated_pow))

# normalise input between 0 and 1
def normalise(vector):
    normed = ((vector - np.min(vector)) / (np.max(vector) - np.min(vector))).reshape(len(vector), 1)
    return normed

# following standard notation: x and y are training, xx and yy are test
x, y = PC_gen(17000) # same number of training points as in paper
xx, yy = PC_gen(2000) # reccomended to keep low as Beta post processing is highly intensive

