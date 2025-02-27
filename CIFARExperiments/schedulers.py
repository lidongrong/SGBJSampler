"""
This script defines some schedulers that can be used in sampling
Scheduler adjusts the step size we take during the sampling process
"""

import math


lr_lambda = lambda k:  1 / (( 1 * (k+1))**0.5)


def csg_mcmc_scheduler(iterations=5000,M=5):
    csg_mcmc = lambda k: 0.5 * (math.cos(math.pi * ((k - 1) % (math.ceil(iterations / M))) / math.ceil(iterations / M))+1)
    return csg_mcmc

def polynomial(p=0.5):
    lr_lambda = lambda k: 1 / ((1 * (k + 1)) ** p)
    return lr_lambda