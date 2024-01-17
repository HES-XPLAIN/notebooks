import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from .custom_dataset import CustomDataset


def train_transform():
    transform = transforms.Compose([
       # transforms.RandomHorizontalFlip(p=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform

def test_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform

def get_dataloaders(batch_size=16):
    data = pd.read_csv("./data/sports.csv")
    data.head()
    data["image_path"] = "./data/" + data["filepaths"]
    lbl = LabelEncoder()
    data["labels_encoded"] = lbl.fit_transform(data["labels"])
    data = data[~data['image_path'].str.endswith('.lnk')]
    df_train = data[data["data set"] == "train"].reset_index(drop=True)
    df_valid = data[data["data set"] == "valid"].reset_index(drop=True)
    df_test = data[data["data set"] == "test"].reset_index(drop=True)
    train_dataset = CustomDataset(df=df_train, transform=train_transform())
    valid_dataset = CustomDataset(df=df_valid, transform=train_transform())
    test_dataset = CustomDataset(df=df_test, transform=test_transform())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['model_state_dict'])
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model

def save_model(epochs, model, optimizer, criterion, name):
    """
    Function to save the trained model to disk.
    """

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"./models/saved_models/{name}.pth")

    print("Model saved :)")


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../models/plots/{name}_accuracy_plot.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../models/plots/{name}_loss_plot.png")


def plot_prediction(model, image_path, transform, class_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    model.eval()

    # Move the input tensor to the device
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Compute the logits
    logits = model(input_tensor)
    probas = torch.nn.functional.softmax(logits, dim=1)

    # Get the top k probabilities and class labels
    top_probas, top_indices = probas.topk(10, dim=1)
    top_probas = top_probas.squeeze().cpu().detach().numpy()
    top_indices = top_indices.squeeze().cpu().detach().numpy()

    # Get the class labels
    class_labels = [class_dict[str(idx)] for idx in top_indices]

    # Plot the horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(10)

    ax.barh(y_pos, top_probas, align='center', color='blue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_labels, fontsize=10)
    ax.invert_yaxis()  # Invert the y-axis to show highest probabilities at the top
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Top 10 Class Probabilities', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix, class_labels):
    # Normalize the confusion matrix to have values between 0 and 1
    normalized_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a colormap with reversed color scheme
    cmap = plt.cm.Reds.reversed()

    # Create the heatmap using matshow
    heatmap = ax.matshow(normalized_matrix, cmap=cmap)

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Set the axis labels
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # Set the title
    ax.set_title('Confusion Matrix')

    # Set the tick labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=6)
    ax.set_yticklabels(class_labels, fontsize=6)

    # Rotate the tick labels if needed
    plt.xticks(rotation=90)

    # Add a dashed line at row 15, column 16
   # ax.axvline(x=16, color='white', linestyle='dashed')
   # ax.axhline(y=15, color='white', linestyle='dashed')

    # Show the plot
    plt.show()
