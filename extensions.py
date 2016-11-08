import logging
import numpy as np
from blocks.extensions import SimpleExtension, TrainingExtension
logger = logging.getLogger('main.utils')


class SaveLog(SimpleExtension):
    def __init__(self, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)
        self.add_condition(('after_training',), self.do)
        self.add_condition(('on_interrupt',), self.do)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, early_stop_var, model, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.early_stop_var = early_stop_var
        self.save_path = save_path
        params_dicts = model.get_parameter_dict()
        self.params_names = params_dicts.keys()
        self.params_values = params_dicts.values()
        self.to_save = {}
        self.best_value = None
        self.add_condition(('after_training',), self.save)
        self.add_condition(('on_interrupt',), self.do)
        self.add_condition(('after_epoch',), self.do)

    def save(self, which_callback, *args):
        to_save = {}
        for p_name, p_value in zip(self.params_names, self.params_values):
            to_save[p_name] = p_value.get_value()
        path = self.save_path + '/trained_params'
        np.savez_compressed(path, **to_save)

    def do(self, which_callback, *args):
        val = self.main_loop.log.current_row[self.early_stop_var]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
            to_save = {}
            for p_name, p_value in zip(self.params_names, self.params_values):
                to_save[p_name] = p_value.get_value()
            path = self.save_path + '/trained_params_best'
            np.savez_compressed(path, **to_save)
        self.main_loop.log.current_row[
            'best_' + self.early_stop_var] = self.best_value

        
class RollsExtension(TrainingExtension):
    """ rolls the cell and state activations between epochs so that first batch gets correct initial activations """
    def __init__(self, shvars):
        self.shvars = shvars
    def before_epoch(self):
        for v in self.shvars:
            v.set_value(np.roll(v.get_value(), 1, 0))

class LearningRateSchedule(TrainingExtension):
    """ Starts decreasing learning rate by a rate after a number of epochs """
    def __init__(self):
        self.epoch_number = 0
    def after_epoch(self):
        self.epoch_number += 1
        if self.epoch_number > decrease_lr_after_epoch:
            learning_rate.set_value(learning_rate.get_value()/lr_decay)
