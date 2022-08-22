import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from src import make_dataset

class CustomDataset(Dataset):
    def __init__(self, data_all, labels, transform=None, target_transform=None):
        self.labels = labels
        self.data_all = data_all
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data_all[idx]
        label = self.labels[idx][0]-1
        if self.transform:
            data = self.transform(data)
            data = data.float()
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

dataset_path = 'D:/Users/shafi/OneDrive/OneDrive - University of Toronto/ISML_22/snn_conversion_2/data/processed/Rat4Training_Fold1.mat'
batch_size = 32

RAT_data = make_dataset.Preprocessing_module('Rat4',1,dataset_path)
train_dataset   = CustomDataset(RAT_data.training_set, RAT_data.training_labels, transforms.ToTensor())
test_dataset    = CustomDataset(RAT_data.test_set, RAT_data.test_labels, transforms.ToTensor())
train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#visualizing the images
labels_map = {
    0: "Dorsiflexion",
    1: "Plantar Flexion",
    2: "Pricking"
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[int(label)])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# testing the loader
import matplotlib.pyplot as plt
# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")