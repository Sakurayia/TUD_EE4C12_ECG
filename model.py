from torch import nn
import torch.nn.functional as F

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(250, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, 15)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = self.layer3(x)
#         return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 256, 5)
        self.layer1 = nn.Linear(256 * (((250 - 4) // 2 - 4) // 2), 120)
        self.layer2 = nn.Linear(120, 84)
        self.layer3 = nn.Linear(84, 15)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x