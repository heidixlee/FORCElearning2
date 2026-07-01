
# set up

import numpy as np

import math

rng = np.random.default_rng(0)

class forceNetwork:
# resevoir
    def __init__(self, N, g, p, alpha, tau, dt):
        # randomly initialize the recurrent weight matrix J (mean = 0, variance = 1/pN) with gain factor
        self.J = rng.normal(0, (g / math.sqrt(p*N)), size = (N,N))
        
        # sparseness parameter applied to the recurrent weight matrix J
        mask = rng.uniform(0, 1, size = (N,N)) < p
        self.J = self.J * mask

        self.tau = tau

        self.dt = dt

        # feedback synapse connection weights
        self.Jgz = rng.uniform(-1, 1, size = N)

        # set internal state of neuron units and initial firing rates
        self.x = rng.normal(0, 0.1, size=N) 
        self.r = np.tanh(self.x)

        # RLS learning matrix
        self.P = np.identity(N) / alpha

        # output weights
        self.w = np.zeros(N)
        # initial output readout
        self.z = 0.0
    
    # time step function, return new x, r, z
    def timestep(self):

        # x over one time step, leaks towards zero,
        # moves positively due to feedback and recurrent activity
        self.x += (self.dt/self.tau) * (-self.x + (self.Jgz * self.z) + (self.J @ self.r))

        # update r
        self.r = np.tanh(self.x)

        # update z
        self.z = self.w @ self.r

        return self.r, self.z
        

    # training function, return new w and P
    def training(self, target):  

        Pr = self.P @ self.r      # correlation activity for each neuron

        rP = self.r @ self.P 

        rPr = self.r @ Pr    # normalization factor

        # RLS update rule for matrix P
        self.P -= np.outer(Pr, rP) / (1 + rPr)

        # calculate error
        error = self.z - target

        self.w -= error * Pr

        self.z = self.w @ self.r

        return self.w

