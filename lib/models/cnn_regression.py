from torch import nn

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ConvolutionalNeuralNetwork,self).__init__()
        self.conv1D_1 = nn.Conv1d(in_channels = input_shape,out_channels=16,kernel_size=1,stride=1)
        self.conv1D_2 = nn.Conv1d(in_channels = 16,out_channels=4,kernel_size=1,stride=1)
        self.linear_1 = nn.Linear(4,1)
        self.drop_out = nn.Dropout(0.4)
        self.selu = nn.SELU()
    

    def forward(self,x):
        out = self.conv1D_1(x)
        out = self.selu(out)
        out = self.drop_out(out)
        out = self.conv1D_2(out)
        out = self.selu(out)
        out = self.drop_out(out)
        out = out.view(-1,4)
        ans = self.linear_1(out)

        return ans

