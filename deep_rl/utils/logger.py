#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import logging

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *


def get_logger(tag='default', log_level=0, save_folder="default"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if tag is not None:
        fh = logging.FileHandler('./log/%s-%s.txt' % (tag, get_time_str()))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return Logger(logger, './tf_log/%s/logger-%s-%s' % (save_folder, tag, get_time_str()), log_level)


class Logger(object):
    def __init__(self, vanilla_logger, log_dir, log_level=0):
        self.log_level = log_level
        self.writer = None
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}
        self.log_dir = log_dir
        self.tracked_scalars = {}

    class tracked_val:
        def __init__(self, obj, retrieve_funct):
            self.a = obj
            self.retrieve_funct = retrieve_funct

        def __init__(self, default_value):
            self.a = default_value
            self.default_value = default_value
            self.retrieve_funct = lambda id: id

        def get(self):
            return self.retrieve_funct(self.a)

        def update_val(self, value):
            self.a = value

        def reset_val(self):
            self.a = self.default_value


    def write_all_tracked_scalars(self, **kwargs):
        tracked_scalars = self.tracked_scalars
        for tag in iter(tracked_scalars):
            self.add_scalar(tag, tracked_scalars[tag].get(), **kwargs)

    def update_log_value(self, tag, value):
        tracked = self.tracked_scalars
        if(tag in tracked.keys()):
            self.tracked_scalars[tag].update_val(value)

    def track_scalar(self, tag, is_by_reference, default_value=0, ref_obj=None, retrieve_funct=lambda obj: obj):
        """Track variable:
            -reference of parent object, with the retrieve funct to get the variable from the parent
            -object is value, with identity retrieve function. In that case it is expected to use "update_log_value" function at each frame.
            this second mechanism is meant for variables that only ever are calculated in local variables.
        """
        # if(object is immutable and retrieve_funct(object) != object):
        #    raise Exception("retrieve_function should be identity for immutable ref. object")
        if is_by_reference:
            self.tracked_scalars[tag] = self.tracked_val(ref_obj, retrieve_funct)
        else:
            self.tracked_scalars[tag] = self.tracked_val(default_value)

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)
