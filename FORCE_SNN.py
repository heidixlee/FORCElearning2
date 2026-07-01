
# set up

import numpy as np

import math

rng = np.random.default_rng(0)

class forceNetwork:
# resevoir
    def __init__(self, N, p, alpha, tau, dt, C, vr, a, b, d, k, vt, vreset, tr, td, vpeak, Q, G, BIAS):

        self.N = N
        self.tau = tau

        self.dt = dt

        # RLS learning matrix
        self.P = np.identity(N) / alpha

        # output weights
        self.w = np.zeros(N)
        # initial output readout
        self.z = 0.0

        # Izhikevich Neuron Initialization
        self.C = C
        self.vr = vr
        self.a = a
        self.b = b
        self.d = d
        self.k = k
        self.vt = vt
        self.vreset = vreset
        self.tr = tr
        self.td = td
        self.vpeak = vpeak

        self.Q = Q
        self.G = G

        self.BIAS = BIAS

        # static recurrent feedback matrix weights -- Scaled by G
        mask = rng.uniform(0, 1, size = (N,N)) < p
        self.OMEGA = G * rng.normal(0, 1, size = (N,N)) < p
        self.OMEGA = (self.OMEGA) * (mask / math.sqrt(N * p)) # normalized and probability mask

        # static internal recurrent weights -- scaled by Q
        self.E = rng.uniform(-1, 1, size = (N,)) * Q

        self.u = np.zeros (N,)    # adaptation variables

        self.v = rng.uniform(low = vr, high = vpeak, size = N)


        self.IPSC = np.zeros (N,)   # final postsynaptice signal 
            # that enters the voltage equation

        self.h = np.zeros (N,)      # integrates JD (weighted spikes) 
            # with decay time td and drives IPSC  

        self.r = np.zeros (N,)     #integrates the raw spike events 
            # and drives the decoder z

        self.JD = np.zeros(N)     # raw spike input

        self.r = np.zeros(N,) # firing rate

    
    # time step function, return new x, r, z
    def timestep(self):

        #sets array of neural activity to zero so that only neurons that spike are considered
        self.JD = np.zeros(self.N)
        # spike_train = np.zeros(self.N)  # which neurons fired

        # continuous evolution through euler's integration

        # constant current I
        I = self.IPSC + self.E * self.z + self.BIAS 

        # update v (internal state)
        v_ = self.v.copy()
        self.v += (self.k * (self.v - self.vr) * (self.v - self.vt) - self.u + I) * (self.dt / self.C)

        # v_ = self.v    # previous t-1 timestep

        # update adaptation variable, uses previous time step
        self.u += self.a * (self.b * (v_ - self.vr) - self.u) * (self.dt/self.tau)

        # identify the neurons that spiked
        # spiked = np.where(self.v >=self.vpeak)[0]
        spiked = self.v >= self.vpeak

        # update the adaptation variable for the neurons that spikes
        self.u[spiked] += self.d

        # reset the neurons that fired to the resting membrane potential
        self.v[spiked] = self.vreset

        if len (spiked) > 0:
            # JD (weight of active neurons by taking the sum of the OMEGA columns of the active neurons)
            self.JD = np.sum(self.OMEGA[:, spiked], axis = 1)

            # spike_train[spiked] = 1.0


        # continuous decay of h
        # self.h += (- self.h / self.tr) * self.dt
        self.h += (1 / (self.tr * self.td)) * self.JD + (- self.h / self.tr) * self.dt

        # update the postsynaptic signal
        self.IPSC += (-self.IPSC / self.td + self.h) * self.dt
        
        # JD drives h
        # self.h += (1 / (self.tr * self.td)) * self.JD

        # update the firing rate
        self.r += (-self.r / self.td + self.h ) * self.dt
        # self.r += spike_train
        self.r += spiked.astype(float)

        # update the output weights
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

