from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np

from lib.models.regression_models.linear_model import LinearModel
from lib.models.regression_models.nn_regression import NeuralNetwork

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
                linear_regression_model.val()
        if self.check_model == 'dlmodel' or self.check_model == 'all':
            print("--RUN DEEP LEARNING--")
            X_train, X_val, Y_train, Y_val = self.data
            Y_train = Y_train[:, [0]]
            Y_val = Y_val[:, [0]]
            assert len(X_train) == len(Y_train)
            self.train(X_train,Y_train,epochs=self.epochs)
            self.eval(X_val,Y_val)

    @classmethod       
    def train(cls, x, y, lr=1e-4, epochs=20,):
        cls.nn_model = NeuralNetwork(input_shape=x.shape[1])
        cls.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(cls.nn_model.parameters(),lr,weight_decay=1e-5)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        print(cls.nn_model)
        print("[EPOCHS]: ", epochs)
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
            print(f"loss RMSE: {loss**(1/2):>7f}")
            print("-------------------------------")
        print("Done!")
    
    def eval(self, x, y):
        test_loss = 0
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        with torch.no_grad():
            for X, label in zip(x,y):
                pred = self.nn_model(X)
                test_loss += self.loss_fn(pred, torch.tensor(label, dtype=torch.float32)).item()
        test_loss /= len(x)
        test_loss = test_loss**(1/2) 
        print(f"Test Error: loss RMSE: {test_loss:>8f} \n")