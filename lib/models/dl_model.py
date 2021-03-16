from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .abstract_model import AbstractModel


class DLModel(AbstractModel):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.nn_model = self.cfg.model.to(self.cfg.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=0, verbose=False)
        self.logger = logger

    def train(self):
        X_train, _, y_train, _ = self.cfg.data

        for e in range(self.cfg.epochs):
            for (X, label) in zip(X_train, y_train):
                # Compute prediction and loss
                self.optimizer.zero_grad()
                pred = self.nn_model(X)
                loss = self.loss_fn(pred, label)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

            loss_eval = self.eval()
            self.logger.info('Epoch [{}/{}] - Train Loss: {:.5f} - Val Loss {:.5f}'.format(e+1, self.cfg.epochs, loss**(1/2), loss_eval))
            self.scheduler.step(-loss_eval)
        self.logger.info('Validation Score: {:.5f}'.format(self.eval()))
    
    def eval(self):
        _, X_val, _, y_val = self.cfg.data
        test_loss = 0
        with torch.no_grad():
            for X, label in zip(X_val, y_val):
                pred = self.nn_model(X)
                test_loss += self.loss_fn(pred, label).item()
        test_loss /= len(X_val)
        return test_loss**(1/2)