import numpy as np
import torch
from torch.optim.optimizer import Optimizer  # , required
from deep_rl.utils.running_avg import RunningAvg, ArrayRunningAvg, MatrixRunningAvg
from deep_rl.utils.torch_utils import to_np


class SimpleSGDLinearDecay(Optimizer):
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
    def __init__(self, params, lr_init=0.1, momentum=0.0, dampening=0, R=0.9, a=1, time_scale=1, logger=None, tag='_', min_lr=0, max_lr=1, **kwargs):
        defaults = dict(lr=lr_init, momentum=momentum, dampening=dampening, R=R)
        super(SimpleSGDLinearDecay, self).__init__(params, defaults)
        self.logger = logger
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.tag = tag
        self.lr_init = lr_init
        self.time_scale = time_scale
        for group in self.param_groups:
            group['lr'] = lr_init
            group['t'] = 0
            group['h'] = -a/lr_init
            group['a'] = a
            logger.update_log_value('lr_' + self.tag, lr_init)
        # time_factor: num steps per update of FDR

    def __setstate__(self, state):
        super(SimpleSGDLinearDecay, self).__setstate__(state)

    # One SGD iteration
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        v_squared = 0.0
        theta_base_norm_sqrd = 0.0
        F_norm_sqrd = 0.0

        for group in self.param_groups:

            # SGD hyperparameters for a group
            group['t'] = group['t'] + self.time_scale
            group['lr'] = group['a']/(group['t'] - group['h'])
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']

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

                F_norm_sqrd += torch.norm(F) ** 2

                # Update velocity and position
                v.mul_(momentum).add_(1.0 - dampening, F)
                v_squared += torch.norm(v) ** 2

                theta.add_(lr, v)

                # Save current v update
                param_state['velocity'] = v

        logger = self.logger
        if logger is not None:
                # dFDR_vals = [('dFDR_' + str(i),elt) for (i, elt) in enumerate(dFDR.shape[0])]
                # OL_vals = [('OL_' + str(i),elt) for (i, elt) in range(OL.shape[0])]

                vals = [("Base_Theta", np.sqrt(theta_base_norm_sqrd)), ('F_norm', np.sqrt(F_norm_sqrd)), ("Velocity", np.sqrt(v_squared)),
                        ('lr', self.param_groups[0]['lr'])]
                if self.tag is not None:
                    tag = self.tag
                    for val in vals:
                        logger.update_log_value(val[0] + "_" + tag, val[1])
                else:
                    for val in vals:
                        logger.update_log_value(val[0], val[1])

    def reduce_lr(self):
        min_lr = self.min_lr
        for group in self.param_groups:
            group['lr'] = group['lr'] * group['R']
            if(group['lr'] < min_lr):
                group['lr'] = min_lr
            self.logger.update_log_value('lr_' + self.tag, group['lr'])

    def augment_lr(self):
        max_lr=self.max_lr
        for group in self.param_groups:
            group['lr'] = group['lr']/group['R']
            if (group['lr'] > max_lr):
                group['lr'] = max_lr
            self.logger.update_log_value('lr_' + self.tag, group['lr'])
