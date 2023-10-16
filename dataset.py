from torch.utils.data import Dataset
import pandas as pd

class ECGDataset(Dataset):
    def __init__(self, path):
        dataframe = pd.read_csv(path, header=None)
        self.data = dataframe.iloc[:, 0:-1]
        self.labels = dataframe.iloc[:, -1:]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y