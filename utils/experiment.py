from flatten_dict import flatten
from comet_ml import Experiment as CometExperiment


class Experiment():
    def __init__(self, api_key=None, **kwargs):
        self._exp = None
        if api_key:
            self._exp = CometExperiment(api_key, log_code=False, **kwargs)

    def log_metric(self, name, value, step=None, epoch=None):
        if self._exp:
            self._exp.log_metric(name, value, step, epoch)

    def log_epoch_end(self, epoch_cnt, step=None):
        if self._exp:
            self._exp.log_epoch_end(epoch_cnt, step=step)

    def log_parameters(self, hp):
        if self._exp:
            self._exp.log_parameters(flatten(hp, reducer='underscore'))

