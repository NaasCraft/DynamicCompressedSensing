"""Definition of model parameters updating algorithms."""
import abc

from dcsamp.parameters import ModelParams


class Updater(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, parameters: ModelParams):
        pass


class EMUpdater(Updater):
    """Expectation Maximization scheme for parameter tuning."""
    def __init__(self):
        pass

    def update(self, parameters: ModelParams):
        pass


class TestUpdater(Updater):
    """Simple pass-through to leave parameters untouched."""
    def update(self, parameters: ModelParams, **kwargs):
        pass
