from torch import nn
import torch
import numpy as np

from lib.models.linear_model import LinearModel
from lib.models.nn_regression import NeuralNetwork

LR = 1e-4
WEIGHT_DECAY = 1e-5

class Runner():
    def __init__(self, cfg, check_ML=True):
        self.models = cfg.ML_models
        self.data = cfg.dataset
        self.epochs = cfg.epochs
        self.check_model = cfg.check_model

    def run(self):
        if self.check_model == 'mlmodel' or self.check_model == 'all':
            for model, name in self.models:
                print(f'================== {name} ==================')

                linear_regression_model = LinearModel(*self.data, model)
                linear_regression_model.train()

        if self.check_model == 'dlmodel' or self.check_model == 'all':
            print("---------------- RUN DEEP LEARNING MODEL ----------------")

            X_train, X_val, Y_train, Y_val = self.data
            assert len(X_train) == len(Y_train) and len(X_val) == len(Y_val) 

            # Train-Eval Part
            self.train(X_train, Y_train, epochs=self.epochs)
            self.eval(X_val, Y_val)

    @classmethod       
    def train(cls, x, y, lr=LR, epochs=20):
        cls.nn_model = NeuralNetwork(input_shape=x.shape[1])
        cls.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(cls.nn_model.parameters(), lr, weight_decay=WEIGHT_DECAY)

        x = x.astype(np.float32)
        x = torch.from_numpy(x)

        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}")
            for (X, label) in zip(x, y):
                # Compute prediction and loss
                optimizer.zero_grad()
                pred = cls.nn_model(X)
                loss = cls.loss_fn(pred, torch.tensor(label, dtype=torch.float32))

                # Backpropagation
                loss.backward()
                optimizer.step()
            print(f"loss RMSE: {loss**(1/2):>8f}")
            print("-------------------------------")
        print("Done!")
    
    def eval(self, x, y):
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        test_loss = 0

        with torch.no_grad():
            for X, label in zip(x, y):
                pred = self.nn_model(X)
                test_loss += self.loss_fn(pred, torch.tensor(label, dtype=torch.float32)).item()
        test_loss /= len(x)
        print(f"Validation Error: loss RMSE: {test_loss**(1/2):>8f} \n")