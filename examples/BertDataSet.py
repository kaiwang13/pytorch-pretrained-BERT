from torch.utils.data import Dataset
import random
import os


class BertDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.set_seed = False

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(os.getpid())
            self.set_seed = True
        return self.data[index]

    def __len__(self):
        return len(self.data)
