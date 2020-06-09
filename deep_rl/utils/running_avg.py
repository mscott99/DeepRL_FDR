import numpy as np
from math import exp


class RunningAvg:
    """Exponential running average"""
    def __init__(self, corr_length, correct_begin=True, init_value = 0):
        self.beta = exp(-1/corr_length)
        self.avg = init_value
        self.correct_begin = correct_begin
        self.n = 0

    def add(self, value):
        beta = self.beta
        n = self.n
        if(self.correct_begin):
            self.avg = 1/(1-beta**(n+1))*(self.avg*(1-beta**n)*beta + (1-beta)*value)
            self.n = n+1
        else:
            self.avg = beta*self.avg +(1-beta)*value

    def get(self):
        return self.avg

class DualRunningAvg:
    """Exponential running average difference, to remove early influence"""
    def __init__(self, corr_length_1, corr_length_2, correct_begin=True, init_value_1 =0, init_value_2=0):
        assert(corr_length_1 > corr_length_2)
        self.first_avg = RunningAvg(corr_length_1,correct_begin, init_value_1)
        self.second_avg= RunningAvg(corr_length_2,correct_begin, init_value_2)
        self.correct_begin = correct_begin

    def weight_of_sum(self, beta, n=0):
        if self.correct_begin:
            return (1 - beta ** (n + 1)) / (1 - beta)
        else:
            return 1/(1-beta)

    def combine_running_avg(self):
        assert(self.first_avg.n == self.second_avg.n)
        n = self.first_avg.n - 1
        beta_1 = self.first_avg.beta
        beta_2 = self.second_avg.beta
        weight_1 = self.weight_of_sum(beta_1, n)
        weight_2 = self.weight_of_sum(beta_2, n)
        if n==0:
            #if statement necessary because weight_1 = weight_2 when n=0, so we have a singularity with no weight on a single element
            return self.first_avg.get()
        return (weight_1 * self.first_avg.get() - weight_2*self.second_avg.get())/(weight_1-weight_2)

    def add(self, value):
        self.first_avg.add(value)
        self.second_avg.add(value)

    def get(self):
        return self.combine_running_avg()
