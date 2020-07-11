from uuid import uuid4

from flatten_dict import flatten

from comet_ml import Experiment as CometExperiment


class Experiment():
    def __init__(self, api_key=None, **kwargs):
        self._exp = None
        self._id = uuid4().hex
        if api_key:
            self._exp = CometExperiment(api_key, log_code=False, **kwargs)
            self._id = self._exp.get_key()

    def log_metric(self, name, value, step=None, epoch=None):
        if self._exp:
            self._exp.log_metric(name, value, step, epoch)

    def log_epoch_end(self, epoch_cnt, step=None):
        if self._exp:
            self._exp.log_epoch_end(epoch_cnt, step=step)

    def log_parameters(self, hp):
        if self._exp:
            self._exp.log_parameters(flatten(hp, reducer='underscore'))

    @property
    def id(self):
        return self._id[:12]


