from torch.utils.data import DataLoader
from dataset import ECGDataset

train_data = ECGDataset("data\\train_data.csv")
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)