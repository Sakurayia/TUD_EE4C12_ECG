import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
from dataset import ECGDataset

train_data = ECGDataset(
    "data\\train_data.csv", 
    target_transform=Lambda(lambda y: torch.zeros(15, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(250, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

data, label = next(iter(train_dataloader))
logits = model(data)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)