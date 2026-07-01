# set up

import numpy as np

rng = np.random.default_rng(0)

import math


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from FORCE_heuns import forceNetwork

# parameters
N = 1000    # number of neuron units

g = 1.6     # strength of recurrent connections

p = 0.1     # connection sparsity parameter

tau = 10    # tau

dt = 0.005  # delta t (change in time)

alpha = 100  # RLS learning rate, used to initialize matrix P (set of learning rates / inverse corelation matric of r)

n_pre = 400
n_training = 5000
n_sim = 900

n_steps = n_pre + n_training + n_sim


# create resevoir, returns J (matric of correlation), Jgz (feedback matrix)
# x (internal state), r (neural activity) and initializes RLS variables
model = forceNetwork(N, g, p, alpha, tau, dt)

# target function
def target_function(t):
    
    period = 120
    omega = math.pi * 2 / period

    function = (period * math.sin(omega * t) 
    + 0.5 * period * math.sin (2 * omega * t)
    + 1/3 * period * math.sin (3 * omega * t) 
    + 0.25 * period * math.sin (4 * omega * t))

    # return normalized target function
    return function / (period * (1 + 0.5 + 1/3 + 0.25))

# simulation
def run_simulation():

    wHist = np.zeros(n_steps)
    zHist = np.zeros(n_steps)
    targetHist = np.zeros(n_steps)

    for i in range (n_steps):
         # advance network by one timestep
        model.r, model.z = model.timestep()

        # convert t to simulation time steps
        t = i * dt

        target = target_function(t)

        if (n_pre <= i < n_pre + n_training):
            # update RLS
            model.w = model.training (target)

        wHist[i] = np.linalg.norm(model.w)
        zHist[i] = model.z
        targetHist[i] = target_function(t)

    return wHist, zHist, targetHist



# plot results
def results(wHist, zHist, targetHist):
    time_axis = np.arange(n_steps) * dt
    
    train_start_time = n_pre * dt
    test_start_time = (n_pre + n_training) * dt

    plt.figure(figsize=(12, 10))

    # PLOT 1 -- Target vs FORCE output
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, targetHist, label="Target Signal", color="black", linewidth=1.5)
    plt.plot(time_axis, zHist, label="FORCE Output", color="red", alpha=0.7)
    plt.axvspan(train_start_time, test_start_time, color='gray', alpha=0.15, label="Training Phase")
    plt.title("Target Function vs. FORCE Network Output")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # PLOT 2 -- ||w||
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, wHist, color="blue", linewidth=2)
    plt.axvspan(train_start_time, test_start_time, color='gray', alpha=0.15)
    plt.title("Weight Vector Norm Progress (Stability Metric)")
    plt.ylabel(r"Norm Magnitude $\|w\|$")
    plt.grid(True, linestyle=":", alpha=0.6)


# Run
wHist, zHist, targetHist = run_simulation()
results(wHist, zHist, targetHist)

# Testing phase RMS error
test_start_idx = n_pre + n_training
training_error = targetHist[test_start_idx:] - zHist[test_start_idx:]
rms_test = np.sqrt(np.mean(training_error ** 2))
print(f"RMS error: {rms_test:.5f}")