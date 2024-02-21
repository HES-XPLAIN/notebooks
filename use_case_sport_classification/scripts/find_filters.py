# global imports

import torch
from torch.utils.data import DataLoader

import json
import pandas as pd
import numpy as np
import torch.nn as nn

from scripts.models import FineTunedVGG, FineTunedEfficientNet
from scripts.custom_dataset import CustomDataset
from scripts.helpers import *

from PIL import Image

#from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FineTunedVGG()
model = model.to(device)
model.eval()

# Specify the map_location argument when loading the model
load_path = "models_weight/VGGFineTuned.pth"
checkpoint = torch.load(load_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# initialize a dataloader containing only image from one class
# for example, we will use the class "air hockey" for this example

def transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def loader_unique_class(class_name):
    data = pd.read_csv("./data/sports.csv")
    data["image_path"] = "./data/" + data["filepaths"]
    lbl = LabelEncoder()
    data["labels_encoded"] = lbl.fit_transform(data["labels"])
    df = data[data["data set"] == "train"].reset_index(drop=True)
    df = df[df["labels"] == class_name].reset_index(drop=True)
    dataset = CustomDataset(df=df, transform=transform())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# store all activations from Conv2D layers
def store_activations(model, dataloader):
    activations = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(get_activation(name, activations))
            hooks.append(hook)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            model(inputs)

    for hook in hooks:
        hook.remove()

    return activations
