import random
import numpy as np
from torchvision import datasets, transforms

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import os
from omnixai_community.data.image import Image as omniImage
from PIL import Image

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.verbose = verbose

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stop")
                return True
        return False

def save_model(epochs, model, optimizer, criterion, name):
    """
    Function to save the trained model to disk.
    """

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"./models_weight/{name}.pth")

    print("Model saved :)")


def plot_confusion_matrix(model, test_loader, device, model_name):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    idx2class = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
    }

    # Assuming val_loader is your validation DataLoader and model is your trained model
    model.eval()
    all_preds = []
    all_labels = []
    model.to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output_probs = model(inputs)
            _, predicted = torch.max(output_probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Plot confusion matrix with ratio values inside the boxes
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    num_classes = len(np.unique(all_labels))
    tick_marks = np.arange(num_classes)
    class_marks = [idx2class[tick] for tick in tick_marks]
    plt.xticks(tick_marks, class_marks, rotation=45)
    plt.yticks(tick_marks, class_marks)

    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # Add ratio values inside the boxes
    thresh = conf_matrix.max() / 2.0  # Threshold for text coloring
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f'{conf_matrix[i, j] / np.sum(conf_matrix[i]):.2f}',
                     horizontalalignment='center',
                     color='white' if conf_matrix[i, j] > thresh else 'black')

    plt.text(num_classes + 1, num_classes - 1, f'Accuracy: {accuracy:.4f}', color='black', fontsize=10)
    plt.text(num_classes + 1, num_classes - 2, f'F1 Score: {f1:.4f}', color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig('confusion_matrix'+model_name+'.png')

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
    plt.savefig(f"./plots/{name}_accuracy_plot.png")

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
    plt.savefig(f"./plots/{name}_loss_plot.png")

def print_rule_nicely(rule, target_class):
    conditions, prediction = rule
    prediction_text = "belongs" if prediction == 1 else "does not belong"
    
    print(f"If", end=" ")
    for i, condition in enumerate(conditions):
        feature, operator, value = condition.split()
        print(f"feature {feature} {operator} {value}", end="")
        if i < len(conditions) - 1:
            print(" and ", end="")
    
    print(f" then the image {prediction_text} to the '{target_class}' class")

def plot_image_and_score(explainer, ax_img, ax_score, img_path, idx2class, class2idx, true_label="Proliferative DR"):

    img = omniImage(Image.open(img_path))
    
    # Generate explanations
    explanations = explainer.explain(img)
    
    # Extract image and score matrices
    expl = explanations.get_explanations(index=0)
    image = expl['image']
    score = expl['scores']
    
    # Get the predicted class
    class_name = idx2class[expl['target_label']]
    
    # Plot the original image
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title(f"Predicted = {class_name}\nTrue = {true_label}")
    
    # Plot the score matrix
    ax_score.imshow(score, cmap='jet')
    ax_score.axis('off')
    ax_score.set_title("Score")

def select_random_images(folder_path, num_images=4):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_files = random.sample(all_files, num_images)
    return [os.path.join(folder_path, f) for f in selected_files]
    

def get_dataloaders(dataset_path: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Specific method to get the dataloaders for the APTOS dataset
    """

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Define transformations for train, validation, and test sets
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the path to your image folder
    data_path = dataset_path

    # Create datasets using ImageFolder
    full_dataset = datasets.ImageFolder(root=data_path)

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Apply respective transforms to each dataset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, pin_memory=True, num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True, num_workers=16, persistent_workers=True)

    return train_loader, val_loader, test_loader
