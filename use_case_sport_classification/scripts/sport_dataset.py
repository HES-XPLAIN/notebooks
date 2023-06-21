import torch
from torch.utils.data import Dataset
from PIL import Image


class SportsData(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.image_paths = self.df.image_path.values.tolist()
        self.labels = self.df.labels.values.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, label
