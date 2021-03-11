from .abstract_model import AbstractModel 
import numpy as np
from sklearn.metrics import mean_squared_error

class LinearModel(AbstractModel):
    def __init__(self, X_train, X_val, Y_train, Y_val, model):
        super(AbstractModel, self).__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = Y_train
        self.y_val = Y_val
        self.model = model

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.predict(self.model, self.X_train, self.y_train, mode='training')
        self.val()

    def val(self):
        self.predict(self.model, self.X_val, self.y_val, mode='validation')

    @staticmethod
    def predict(model, x, y, mode='training'):
        y_predict = model.predict(x)
        print(f'Root Mean squared error {mode}: {np.sqrt(mean_squared_error(y, y_predict)):.2f}')