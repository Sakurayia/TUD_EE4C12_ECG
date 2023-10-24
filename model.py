from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear_relu_stack = nn.Sequential(
        #    nn.Linear(250, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 15),
        #)
        self.layer1 = nn.Linear(250, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 15)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x