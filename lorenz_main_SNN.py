# set up

import numpy as np

rng = np.random.default_rng(0)

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from FORCE_lorenz_SNN import forceNetwork

# model and RLS parameters
N = 1000    # number of neuron units
p = 0.1     # connection sparsity parameter
tau = 0.2   # tau 
dt = 0.001  # delta t (change in time)
alpha = 100  # RLS learning rate, used to initialize matrix P (set of learning rates / inverse corelation matric of r)


# number of steps
n_pre = 3600
n_training = 20000
n_sim = 3600

n_steps = n_pre + n_training + n_sim

# Izhikevich Neuron Parameters
C = 250     # capatitance

vr = -60    # resting membrane potential

a = 0.01    # reciprocal of adaptation

vpeak = 30

b = -2      # resonance model properties

d = 200     # adaptation current

k = 2.5     # action potential half width

vt = vr + 40 - (b/k)    # threshold potential

vreset = -65

Q = 5 * 10 ** 3

G = 5 * 10 ** 3

tr = 2      # synaptic rise time

td = 20     # synaptice decay time

BIAS = 1000     # constant 




# create resevoir, returns J (matric of correlation), Jgz (feedback matrix)
# x (internal state), r (neural activity) and initializes RLS variables
model = forceNetwork(N, p, alpha, tau, dt, C, vr, a, b, d, k, vt, vreset, tr, td, vpeak, Q, G, BIAS)


def lorenz_deriv(state, p=28, o=10, b=8/3):
    x, y, z = state
    dx = o * (y - x)
    dy = x * (p - z) - y
    dz = (x * y) - (b * z)
    return np.array([dx, dy, dz])


#fourth order lorenz (RK4)
def target_func(n_steps, dt, p=28, o=10, b=8/3):
    xyz = np.zeros((n_steps, 3))
    xyz[0] = [0.1, 0.0, 0.0]
    
    for i in range(n_steps - 1):
        state = xyz[i]
        k1 = lorenz_deriv(state, p, o, b)
        k2 = lorenz_deriv(state + dt/2 * k1, p, o, b)
        k3 = lorenz_deriv(state + dt/2 * k2, p, o, b)
        k4 = lorenz_deriv(state + dt * k3, p, o, b)
        
        xyz[i+1] = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return xyz



# simulation
def run_simulation():

    wHist = np.zeros(n_steps)
    zHist = np.zeros((n_steps, 3))
    targetHist = np.zeros((n_steps, 3))
    lorenz = target_func(n_steps, dt)
    lorenz = lorenz / 40.0 # normalization to keep the function between -1 and 1

    for i in range (n_steps):
         # advance network by one timestep
        model.r, model.z = model.timestep()

        target = lorenz[i] 

        if (n_pre <= i < n_pre + n_training):
            # update RLS
            model.w = model.training (target)

        wHist[i] = np.linalg.norm(model.w)
        zHist[i] = model.z
        targetHist[i] = target

    return wHist, zHist, targetHist


def plot_attractor(zHist, targetHist, test_start_idx):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(targetHist[test_start_idx:, 0],
              targetHist[test_start_idx:, 1],
              targetHist[test_start_idx:, 2],
              color='black', linewidth=0.5)
    ax1.set_title("Target Lorenz Attractor")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(zHist[test_start_idx:, 0],
              zHist[test_start_idx:, 1],
              zHist[test_start_idx:, 2],
              color='red', linewidth=0.5)
    ax2.set_title("Network Output Attractor")

    plt.tight_layout()
    plt.savefig("attractor_comparison.png", dpi=150)

    plt.figure(figsize=(12, 10))

# plot results
def results(wHist, zHist, targetHist):


    time_axis = np.arange(n_steps) * dt
    
    train_start_time = n_pre * dt
    test_start_time = (n_pre + n_training) * dt

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

    plt.tight_layout()
    plt.savefig("force_results.png", dpi=150)   # ← this line was missing
    plt.close()


# Run
wHist, zHist, targetHist = run_simulation()
results(wHist, zHist, targetHist)

# Testing phase RMS error
test_start_idx = n_pre + n_training
plot_attractor(zHist, targetHist, test_start_idx)
training_error = targetHist[test_start_idx:] - zHist[test_start_idx:]
rms_test = np.sqrt(np.mean(training_error ** 2))

# Short window correlation - only look at the first ~50-100 steps post training
# before chaotic divergence dominates
short_window = 200  # adjust based on your dt; try ~1 time unit worth of steps

short_target = targetHist[test_start_idx : test_start_idx + short_window]
short_output = zHist[test_start_idx : test_start_idx + short_window]

correlations = []
for dim in range(3):
    corr = np.corrcoef(short_target[:, dim], short_output[:, dim])[0, 1]
    correlations.append(corr)
    print(f"Dimension {['x','y','z'][dim]} correlation: {corr:.4f}")

print(f"Average correlation: {np.mean(correlations):.4f}")
print(f"RMS error: {rms_test:.5f}")
