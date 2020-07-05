'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the LICENSE_fb file in the root directory of this source tree.
    
    PyTorch implementation of the scheduler based on the fluctuation-dissipation relation described in:
    Sho Yaida, "Fluctuation-dissipation relations for stochastic gradient descent," ICLR 2019.
'''
import numpy as np
import torch
from torch.optim.optimizer import Optimizer  # , required
from deep_rl.utils.running_avg import RunningAvg, ArrayRunningAvg, MatrixRunningAvg
from deep_rl.utils.torch_utils import to_np


class FDRCtrlFreak(Optimizer):
    '''
    Arguments:
        lr_init (float, optional, default=0.1): initial learning rate.
        
        momentum (float, optional, default=0.0): momentum.
        
        dampening (float, optional, default=0.0): dampening.
        
        weight_decay (float, optional, default=0.001): weight_decay.
            If weight_decay=0, SGD sampling typically does not reach equilibrium. Hence it should be set away from 0.
            
        t_adaptive (int, optional, default=1000 SGD iterations): how frequently one checks proximity to equilibrium.
        
        X (float, optional, default=0.01): how stringently one imposes the equilibrium condition.
            For example, the default value 0.01 means that the fluctuation-dissipation relation needs to be satisfied within one-percent error.
            
        Y (float, optional, default=0.9): learning rate decreases by *(1-Y) once SGD sampling is deemed close to equilibrium.
            For example, the default value 0.9 implies the ten-fold decrease in the learning rate.
    '''

    # Setting up hyperparameters
    def __init__(self, params, lr_init=0.1, momentum=0.0, dampening=0, low_callback = lambda:None, high_callback=lambda:None,
                 X_low=0.00, X_high=1e8, R=0.9, logger=None, tag=None, time_factor=1, sceptic_period = 0, min_baseline_length = 1.0,
                 max_baseline_length=1e6, min_FDR_length = 1.0, max_FDR_length=1e6, run_avg_base=2.0,
                 low_count_threshold=1,high_count_threshold=1, high_ratio=0.5, max_lr=1, min_lr= 1e-6, **kwargs):
        defaults = dict(lr=lr_init, momentum=momentum, dampening=dampening, X_low=X_low, X_high=X_high, R=R)
        super(FDRCtrlFreak, self).__init__(params, defaults)
        self.logger = logger
        self.low_callback=low_callback
        self.high_callback=high_callback
        self.tag = tag
        self.lr_init = lr_init
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.time_factor = time_factor
        self.min_baseline_length = min_baseline_length
        self.max_baseline_length = max_baseline_length
        self.min_FDR_length = min_FDR_length
        self.max_FDR_length = max_FDR_length
        self.run_avg_base = run_avg_base
        self.low_count_threshold = low_count_threshold
        self.high_count_threshold = high_count_threshold
        self.high_ratio = high_ratio
        self.reset_stats()
        self.change_learner = False #signal
        self.sceptic_period = sceptic_period
        #self.reduce_num = 1 #number of times to reduce lr before passing the ball
        #self.reduced_so_far = 0 # number of time lr was reduced

        for group in self.param_groups:
            group['lr'] = lr_init
            logger.update_log_value('lr_' + self.tag, lr_init)
        #time_factor: num steps per update of FDR

    def __setstate__(self, state):
        super(FDRCtrlFreak, self).__setstate__(state)

    # One SGD iteration
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # SGD hyperparameters for a group
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']



            # Allocating memory for observables in the first fluctuation-dissipation relation
            #if 'OLs' not in group:
            #    # Observables in the first fluctuation-dissipation relation
            #    group['OLs'] = list()
                #group['ORs'] = list()
                #group['t'] = 0
                #group['dORbar'] = RunningAvg(int(100/self.time_factor))
                #group['dOLbar'] = RunningAvg(int(100/self.time_factor))


            # Actual SGD step
            # F = -(gradient of loss)
            # v = momentum*v+(1-dampening)*F
            # theta = theta+lr*v where
            v_squared = 0.0
            theta_base_norm_sqrd = 0.0
            F_norm_sqrd = 0.0
            OL = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                theta = p.data
                F = -p.grad.data

                # Create velocity degrees of freedom
                param_state = self.state[p]
                if 'velocity' not in param_state:
                    v = param_state['velocity'] = torch.zeros_like(theta)
                else:
                    v = param_state['velocity']

                # Create baseline
                if 'baseline' not in param_state:
                    #baseline = param_state['baseline'] = torch.zeros_like(theta)
                    baseline = param_state['baseline'] = ArrayRunningAvg(exp_base=self.run_avg_base, low_bound=int(self.min_baseline_length/self.time_factor), high_bound=int(self.max_baseline_length/self.time_factor), correct_begin=True,
                                                                    init_value=torch.zeros_like(theta.view(-1)))
                else:
                    baseline = param_state['baseline']

                F_norm_sqrd += torch.norm(F) ** 2
                theta_tilde = theta.view(-1).unsqueeze(0) - baseline.get()
                OL += -torch.tensordot(F.view(-1), theta_tilde,dims=[[0],[1]])
                #OL is a matrix. the first dimention is the different lengths of running average of OL. The second dimention is the length of baselines

                # Update velocity and position
                v.mul_(momentum).add_(1.0 - dampening, F)
                v_squared += torch.norm(v) ** 2

                baseline.add(value_for_all=theta.view(-1)) # update baseline a step late to minimize correlation
                theta.add_(lr, v)

                # Save current v update
                param_state['velocity'] = v


            # Record OL and OR
            OR = 0.5* lr * ((1.0 + momentum) / (1.0 - dampening)) * v_squared
            # normalize both by learning rate
            OL = OL/lr
            OR = OR/lr

            group['t'] += 1

            group['dOLbar'].add(first_dim_vals=OL) # we have individual values for OL because of the different baselines in OL
            group['dORbar'].add(value_for_all=OR)

            dFDR = np.abs(to_np(group['dOLbar'].get()/group['dORbar'].get().unsqueeze(0) - 1))
            #dFDR = np.array([np.abs((Ol.get() / (Or.get())) - 1.0) for (Ol,Or) in zip(group['dOLbar'], group['dORbar'])])


            logger = self.logger
            if logger is not None:
                #dFDR_vals = [('dFDR_' + str(i),elt) for (i, elt) in enumerate(dFDR.shape[0])]
                #OL_vals = [('OL_' + str(i),elt) for (i, elt) in range(OL.shape[0])]

                vals = [("OR", OR), ("Base_Theta", np.sqrt(theta_base_norm_sqrd)),('F_norm', np.sqrt(F_norm_sqrd)), ('dFDR', dFDR[3,3].item())]
                vals = vals #+ dFDR_vals + OL_vals
                if self.tag is not None:
                    tag = self.tag
                    for val in vals:
                        logger.update_log_value(val[0] + "_" + tag, val[1])
                else:
                    for val in vals:
                        logger.update_log_value(val[0], val[1])

            #count the number of time each average was satisfied
            small = (dFDR < group['X_low']).astype(int)
            big = (dFDR > group['X_high']).astype(int)
            self.low_counter = self.low_counter*small + small
            self.high_counter = self.high_counter*big + big

            logger.update_log_value('max_low_count', np.max(self.low_counter))
            logger.update_log_value('max_high_count', np.max(self.high_counter))
            #if any avg is below threshold, good enough
            if (self.low_counter >= self.low_count_threshold).any() and group['t']*self.time_factor > self.sceptic_period:
                self.reduce_lr_with_check()
                group['t']=0
            #if majority of avg is above threshold, go up
            elif np.sum(self.high_counter >= self.high_count_threshold) > self.high_counter.size*self.high_ratio and group['t'] * self.time_factor > self.sceptic_period:
                self.augment_lr_with_check()
                group['t']=0
    def reset_stats(self):
        for group in self.param_groups:
            # Purge running record of observables
            group['OLs'] = list()
            group['ORs'] = list()
            group['dORbar'] = ArrayRunningAvg(exp_base=self.run_avg_base, low_bound=int(self.min_FDR_length/self.time_factor), high_bound=int(self.max_FDR_length/self.time_factor))
            group['dOLbar'] = MatrixRunningAvg(exp_base=self.run_avg_base, first_low_bound=int(self.min_baseline_length/self.time_factor) ,
                              first_high_bound=int(self.max_baseline_length/self.time_factor) ,low_bound=int(self.min_FDR_length/self.time_factor), high_bound=int(self.max_FDR_length/self.time_factor))
            self.low_counter = np.zeros_like(to_np(group['dOLbar'].get()))
            self.high_counter = np.zeros_like(to_np(group['dOLbar'].get()))
            # Reset time
            group['t'] = 0

    def reduce_lr_with_check(self):
        self.reduce_lr()
        called_callback = False
        for group in self.param_groups:
            if group['lr'] < self.min_lr:
                group['lr'] = self.min_lr
                if not called_callback:
                    self.low_callback()
                    called_callback = True
        self.logger.update_log_value('lr_' + self.tag, group['lr'])

    def augment_lr_with_check(self):
        self.augment_lr()
        called_callback = False
        for group in self.param_groups:
            if group['lr'] > self.max_lr:
                group['lr'] = self.max_lr
                if not called_callback:
                    self.high_callback()
                    called_callback = True
        self.logger.update_log_value('lr_' + self.tag, group['lr'])

    def reduce_lr(self):
        for group in self.param_groups:
            group['lr'] = group['lr'] * group['R']
            self.logger.update_log_value('lr_' + self.tag, group['lr'])

    def augment_lr(self):
        for group in self.param_groups:
            group['lr'] = group['lr']/group['R']
            self.logger.update_log_value('lr_' + self.tag, group['lr'])

