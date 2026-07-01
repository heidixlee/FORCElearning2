
# set up

import numpy as np

import math

rng = np.random.default_rng(0)

class forceNetwork:
# resevoir
    def __init__(self, N, g, p, alpha, tau, dt, Q):
        # randomly initialize the recurrent weight matrix J (mean = 0, variance = 1/pN) with gain factor
        self.J = rng.normal(0, (g / math.sqrt(p*N)), size = (N,N))
        
        # sparseness parameter applied to the recurrent weight matrix J
        mask = rng.uniform(0, 1, size = (N,N)) < p
        self.J = self.J * mask

        self.tau = tau

        self.dt = dt

        self.Q = Q

        # feedback synapse connection weights
        self.Jgz = rng.uniform(-1, 1, size = (N,3)) * self.Q

        # set internal state of neuron units and initial firing rates
        self.x = rng.normal(0, 0.1, size = (N,)) 
        self.r = np.tanh(self.x)

        # RLS learning matrix
        self.P = np.identity(N) / alpha

        # output weights
        self.w = np.zeros((N,3))
        # initial output readout
        self.z = np.zeros(3)
    
    # time step function, return new x, r, z
    def timestep(self):

        # x over one time step, leaks towards zero,
        # moves positively due to feedback and recurrent activity

        # slope (current rate of change)
        rate1 = (-self.x + (self.Jgz @ self.z) + (self.J @ self.r)) / self.tau

        # predict the next point with this slope
        predicted_x = self.x + rate1 * self.dt

        # predict the slope @ the predicted point
        predicted_r = np.tanh(predicted_x)
        predicted_z = predicted_r @ self.w
        rate2 = (-predicted_x + self.Jgz @ predicted_z + self.J @ predicted_r) / self.tau
 
        # calculate the "midpoint" / average between the two slopes to find the actual next step to take
        self.x += ((rate1 + rate2) / 2) * self.dt

        # update r
        self.r = np.tanh(self.x)

        # update z
        self.z = self.w.T @ self.r

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

        self.w -= np.outer(Pr, error)

        self.z = self.w.T @ self.r

        return self.w

