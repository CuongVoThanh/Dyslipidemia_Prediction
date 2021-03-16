from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def train(self):
        """ Train Phrase """
        raise NotImplementedError()

    @abstractmethod
    def eval(self):
        """ Validation Phrase """
        raise NotImplementedError()