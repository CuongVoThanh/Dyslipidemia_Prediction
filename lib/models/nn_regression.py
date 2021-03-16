from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
        )   
    
    def forward(self,x):
        res = self.linear_relu_stack(x)
        return res