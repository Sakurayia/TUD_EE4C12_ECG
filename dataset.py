from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        dataframe = pd.read_csv(path, header=None)
        self.data = np.array(dataframe.iloc[:, 0:-1])
        self.labels = np.array(dataframe.iloc[:, -1:]).astype('int')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label
