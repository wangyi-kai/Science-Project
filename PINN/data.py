from libs import *
from equation import *
from net import *

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, label):
        self.data = data_root
        self.label = label

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.data)

