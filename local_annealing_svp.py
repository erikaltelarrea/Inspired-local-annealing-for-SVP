import torch
import torch.nn as nn
import time
from math import pi
import random 
import numpy as np
from matplotlib import pyplot as plt
import itertools

class Lqa_svp(nn.Module):
    """
    param couplings_bin: square symmetric torch array encoding the 2-body interactions of the Hamiltonian
    param fields_bin: torch array encoding the 1-body interactions of the Hamiltonian
    param identity: constant in the Hamiltonian
    param r, s: penalization hyperparameters
    """
    def __init__(self, couplings_bin, fields_bin, identity, qubits, qudits, r, s):
        super(Lqa_svp, self).__init__()

        self.couplings_bin = couplings_bin
        self.fields_bin = fields_bin
        self.identity = identity
        self.n = couplings_bin.shape[0]
        self.qubits = qubits
        self.qudits = qudits
        self.energy = 0.
        self.config = torch.zeros([self.n, 1])
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, 1])
        self.weights = torch.ones([self.n])
        self.r = r
        self.s = s

        # Move data to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.couplings_bin = self.couplings_bin.to(self.device)
        self.fields_bin = self.fields_bin.to(self.device)
        self.identity = torch.tensor(identity).to(self.device)
        self.r = torch.tensor(r).to(self.device)
        self.s = torch.tensor(s).to(self.device)

    def schedule_1(self, i, N):
        """Computes the first annealing schedule at point i"""
        return (i/N)**1.0 

    def schedule_2(self, i, N):
        """Computes the second annealing schedule at point i"""
        return (i/N)**3.8 
    
    def energy_ising(self, config):
        """Computes the energy of the system given a configuration.
            * config: product state to compute the energy """
        config = config.to(self.device)
        energy_1 = torch.matmul(torch.matmul(self.couplings_bin, config), config) + torch.matmul(self.fields_bin, config) + self.identity 
        energy = energy_1 + self.r * torch.exp(-self.s * energy_1) 

        return energy

    def energy_full(self, t_1, t_2, g):
        """Computes the cost function value that corresponds to the annealing Hamiltonian.
            * t_1: first annelaing schedule
            * t_2: second annealing schedule
            * g: gamma in the article, strength of the H_z"""
        config = torch.tanh(self.weights)*pi/2
        ez = self.energy_ising(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t_2*ez*g - (1-t_1)*ex)


    def minimise(self,
                 step,  
                 N, 
                 g,
                 f):
        """Minimizes the cost function using stochastic gradient descent (SGD). The initial configuration 
        corresponds to the ground state of the initial Hamiltonian which is the plus 
        state. In the end, the outputs are post-processed such that we obtain integers that 
        span the linear combination of the shortest vector (using Bin-encoded qudits).
            * step: learning rate
            * N: number of steps
            * g: gamma in the article
            * f: constant that multiplies the random weight initialization"""
        self.weights = (2 * torch.rand([self.n]) - 1) * f 

        self.weights.requires_grad=True
        optimizer = torch.optim.SGD([self.weights], lr=step, momentum=0.9989)

        for i in range(N):
            t_1 = self.schedule_1(i, N)
            t_2 = self.schedule_2(i, N)
            energy = self.energy_full(t_1, t_2, g)
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

        self.z_string = torch.sign(self.weights.detach())

        self.energy = float(self.energy_ising(self.z_string))

        vect = []
        for i in range(self.qudits):
            num = -1/2
            for j in range(self.qubits):
                num += -self.z_string[i * self.qubits + j] * (2 ** (j-1))
            vect.append(num)
        self.config = [int(y) for y in vect]

        return self.energy, self.config, self.z_string