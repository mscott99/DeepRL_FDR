import numpy as np
from math import exp
import math
import torch


class RunningAvg:
    """Exponential running average"""
    def __init__(self, corr_length, correct_begin=True, init_value = 0):
        self.beta = exp(-1/corr_length)
        self.corr_length = corr_length
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


class ArrayRunningAvg:
    def __init__(self, exp_base=2, low_bound=1000, high_bound=1e6, **kwargs):
        start = math.ceil(math.log(low_bound, exp_base))
        stop = math.ceil(math.log(high_bound, exp_base))
        self.avgs = [RunningAvg(exp_base**i, **kwargs) for i in range(start, stop+1)]
        if True:
            pass

    def __iter__(self):
        return iter(self.avgs)

    def get(self):
        result = list(map(lambda obj: torch.tensor(obj.get()).unsqueeze(0), self.avgs))
        return torch.cat(result)

    def add(self, value_for_all=None, indiv_values=None):
        if value_for_all is not None:
            for elt in self.avgs:
                elt.add(value_for_all)
        elif indiv_values is not None:
            for (elt,val) in zip(self.avgs, indiv_values):
                elt.add(val)

class MatrixRunningAvg:
    def __init__(self, exp_base=2, first_low_bound=1000, first_high_bound=1e6, **kwargs):
        start = math.ceil(math.log(first_low_bound, exp_base))
        stop = math.ceil(math.log(first_high_bound, exp_base))
        self.avgs = [ArrayRunningAvg(exp_base=exp_base, **kwargs) for i in range(start, stop+1)]
        #we include the end
        if True:
            pass

    def __iter__(self):
        return iter(self.avgs)

    def get(self):
        result = list(map(lambda obj: obj.get().unsqueeze(0), self.avgs))
        return torch.cat(result)

    def add(self, second_dim_vals=None, first_dim_vals=None):
        if second_dim_vals is not None:
            for elt_list in self.avgs:
                for val, elt in zip(second_dim_vals, elt_list):
                    elt.add(val)
        elif first_dim_vals is not None:
            for value, elt in zip(first_dim_vals, self.avgs):
                elt.add(value_for_all=value)
