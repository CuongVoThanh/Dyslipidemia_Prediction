from torch import nn

FLATTEN_SHAPE = 128

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ConvolutionalNeuralNetwork,self).__init__()
        self.conv1D_1 = nn.Conv1d(in_channels=input_shape, out_channels=256, kernel_size=1, stride=1)
        self.linear_1 = nn.Linear(FLATTEN_SHAPE, 1)
        self.drop_out = nn.Dropout(0.2)
        self.selu = nn.SELU()

    def forward(self, x):
        out = self.conv1D_1(x)
        out = self.selu(out)
        out = self.drop_out(out)
        out = out.view(-1, FLATTEN_SHAPE)
        ans = self.linear_1(out)

        return ans

