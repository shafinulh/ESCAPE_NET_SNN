from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Modifying torch built-in dataset class to be capable of handling the RAT dataset"""

    def __init__(self, data_all, labels, transform=None, target_transform=None):
        self.labels = labels
        self.data_all = data_all
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data_all[idx]
        label = self.labels[idx][0] - 1
        if self.transform:
            data = self.transform(data)
            data = data.float()
        if self.target_transform:
            label = self.target_transform(label)
        return data, label
