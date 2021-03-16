from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np

from .abstract_model import AbstractModel 

SEED = 42


class MLModel(AbstractModel):
    def __init__(self, X_train, X_val, Y_train, Y_val, model, logger):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = Y_train
        self.y_val = Y_val
        self.model = model
        self.logger = logger

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        train_score = self.predict(self.model, self.X_train, self.y_train)
        eval_score = self.eval()
        return train_score, eval_score

    def eval(self):
        return self.predict(self.model, self.X_val, self.y_val)

    @classmethod
    def evaluate_model_by_kfold(cls, X, y, model, kfold, logger):
        """
            Utilize K-fold to train the model
            Return: The best model with respect to get minimum RMSE score of that fold
        """
        cv = KFold(n_splits=kfold, shuffle=True, random_state=SEED)
        
        best_score = float('inf')
        for train_index, val_index in cv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            lr_model = cls(X_train, X_val, y_train, y_val, model, logger)
            _, eval_score = lr_model.train()

            if best_score > eval_score:
                best_score = eval_score
                lr_model.score = eval_score
                best_model = lr_model 

        return best_model

    @staticmethod
    def predict(model, X, y):
        y_predict = model.predict(X)
        score = np.sqrt(mean_squared_error(y, y_predict))
        return score

    @staticmethod
    def evaluate_model_by_cv(X, y, cv, model):
        """ Evaluate by any cross validation function """
        scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        return -np.max(scores)