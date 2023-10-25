from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from terminaltables import AsciiTable



class ECGDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None, class_name=[]):
        dataframe = pd.read_csv(path, header=None)
        self.data = np.array(dataframe.iloc[:, 0:-1])
        self.labels = np.array(dataframe.iloc[:, -1:]).astype(np.int64)
        self.transform = transform
        self.target_transform = target_transform
        self.class_name = class_name

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
    
    def compute_class_num(self, table_print=True):
        cls_counter = [0 for _ in range(len(self.class_name))]

        for label in self.labels:
            cls_counter[label[0]] += 1

        if table_print:
            class_table_data = [['Class', 'Count']]

            for cls_name, cnt in zip(self.class_name, cls_counter):
                class_table_data.append([cls_name, cnt])
            class_table_data.append(['total', sum(cls_counter)])

            table = AsciiTable(class_table_data)
            print('\n' + table.table)

        return cls_counter
    
    def compute_class_weights(self):
        cls_counter = self.compute_class_num(table_print=False)
        total = sum(cls_counter)
        #max_ = max(cls_counter)
        cls_freq = [cnt / total for cnt in cls_counter]
        cls_weights = [1.0 / freq for freq in cls_freq]

        return cls_weights
