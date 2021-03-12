from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def train(self):
        """ Train Phrase """
        raise NotImplementedError()

    @abstractmethod
    def val(self):
        """ Validation Phrase """
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        """ Prediction Result """
        raise NotImplementedError()