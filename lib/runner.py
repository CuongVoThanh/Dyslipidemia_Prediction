from torch import nn
import torch

from lib.models.linear_model import LinearModel
from lib.models.nn_regression import NeuralNetwork

LR = 1e-4
WEIGHT_DECAY = 1e-5

class Runner():
    def __init__(self, cfg):
        self.models = cfg.ML_models
        self.data = cfg.dataset
        self.epochs = cfg.epochs
        self.check_model = cfg.check_model
        self.device = cfg.device

    def run(self):
        if self.check_model == 'mlmodel' or self.check_model == 'all':
            for model, name in self.models:
                print(f'================== {name} ==================')

                linear_regression_model = LinearModel(*self.data, model)
                linear_regression_model.train()

        if self.check_model == 'dlmodel' or self.check_model == 'all':
            print("---------------- RUN DEEP LEARNING MODEL ----------------")
            print(f"DEVICE-IN-USE: {self.device}")
            
            X_train, X_val, Y_train, Y_val = [torch.tensor(data, dtype=torch.float32).to(self.device) for data in self.data]
            assert len(X_train) == len(Y_train) and len(X_val) == len(Y_val) 

            # Train-Eval Part
            self.train(X_train, Y_train, self.device, epochs=self.epochs)
            self.eval(X_val, Y_val)

    @classmethod       
    def train(cls, x, y, device='cpu', lr=LR, epochs=20):
        cls.nn_model = NeuralNetwork(input_shape=x.shape[1]).to(device)
        cls.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(cls.nn_model.parameters(), lr, weight_decay=WEIGHT_DECAY)
        for e in range(epochs):
            print(f"Epoch {e+1}/{epochs}")
            for (X, label) in zip(x, y):
                # Compute prediction and loss
                optimizer.zero_grad()
                pred = cls.nn_model(X)
                loss = cls.loss_fn(pred, label)

                # Backpropagation
                loss.backward()
                optimizer.step()
            print(f"loss RMSE: {loss**(1/2):>8f}")
            print("-------------------------------")
        print("Done!")
    
    def eval(self, x, y):
        test_loss = 0
        with torch.no_grad():
            for X, label in zip(x, y):
                pred = self.nn_model(X)
                test_loss += self.loss_fn(pred, label).item()
        test_loss /= len(x)
        print(f"Validation Error: loss RMSE: {test_loss**(1/2):>8f} \n")