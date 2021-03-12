from .abstract_model import AbstractModel 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np

SEED = 42


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

    @classmethod
    def evaluate_model_by_kfold(cls, X, y, model, kfold):
        cv = KFold(n_splits=kfold, shuffle=True, random_state=SEED)
        score = cls.evaluate_model_by_cv(X, y, cv, model)
        print(f'Root Mean Squared Error {kfold}-fold: {score:.2f}')

    @staticmethod
    def predict(model, x, y, mode='training'):
        y_predict = model.predict(x)
        print(f'Root Mean Squared Error {mode}: {np.sqrt(mean_squared_error(y, y_predict)):.2f}')

    @staticmethod
    def evaluate_model_by_cv(X, y, cv, model):
        scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        return -np.max(scores)