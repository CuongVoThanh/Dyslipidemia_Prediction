from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np

from lib.models.regression_models.linear_model import LinearModel
from lib.models.regression_models.nn_regression import NeuralNetwork

def train_loop(x_train, y_train, model, loss_fn, optimizer):
    for (X, y) in zip(x_train, y_train):
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        # print("[DEBUG]: ",pred, y)
        loss = loss_fn(pred, torch.tensor(y, dtype=torch.float32))

        # Backpropagation
        loss.backward()
        optimizer.step()
    print(f"loss MSE: {loss**(1/2):>7f}")

class Runner():
    def __init__(self, cfg, check_ML=True):
        self.check_ML = check_ML
        self.models = cfg.ML_models
        self.data = cfg.dataset
        self.epochs = cfg.epochs

    def run(self):
        if self.check_ML == True:
            for model, name in self.models:
                print(f'================== {name} ==================')
                linear_regression_model = LinearModel(*self.data, model)
                linear_regression_model.val()
        else:
            print("--RUN DEEP LEARNING--")
            X_train, X_val, Y_train, Y_val = self.data
            Y_train = Y_train[:, [0]]
            Y_val = Y_val[:, [0]]
            assert len(X_train) == len(Y_train)
            self.train(X_train,Y_train,self.epochs)
            self.eval(X_val,Y_val)

    @classmethod       
    def train(cls, x, y, batch_size=64, lr=1e-3, epochs=20):
        cls.nn_model = NeuralNetwork(input_shape=x.shape[1])
        cls.loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(cls.nn_model.parameters(),lr)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}")
            train_loop(x, y, cls.nn_model, cls.loss_fn, optimizer)
            print("-------------------------------")
        print("Done!")
    
    def eval(self, x, y):
        test_loss = 0
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        with torch.no_grad():
            for X, y in zip(x,y):
                pred = self.nn_model(X)
                test_loss += self.loss_fn(pred, torch.tensor(y, dtype=torch.float32)).item()
        test_loss /= len(x)
        test_loss = test_loss**(1/2) 
        print(f"Test Error: loss MSE: {test_loss:>8f} \n")