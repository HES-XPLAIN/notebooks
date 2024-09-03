import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import VGGAptos
from helpers import get_dataloaders, plot_confusion_matrix, EarlyStopper


def train(model, train_loader, optimizer, criterion, device):
    """This function performs one epoch of training.

    Args:
        model (torch.nn.Module): VGG model to fine-tune.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.modules.loss._Loss): Loss function used during training.
        device (torch.device): Device on which to perform training, e.g., 'cuda' or 'cpu'.

    Returns:
        tuple: A tuple containing training loss (float) and accuracy (float) per epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc='Training', position=0, leave=True)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output_probs = model(inputs)
        loss = criterion(output_probs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(output_probs, 1)
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        # Update the progress bar
        progress_bar.set_postfix({'loss': total_loss / len(train_loader),
                                  'accuracy': 100. * total_correct / total_samples})

    return total_loss / len(train_loader), 100. * total_correct / total_samples


def validation( model, val_loader, criterion, device):
    """This function do one epoch of validation.

    Args:
        model (torch.nn.Module): VGG model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.modules.loss._Loss): Loss function used for validation.
        device (torch.device): Device on which to perform validation, e.g., 'cuda' or 'cpu'.

    Returns:
        tuple: A tuple containing validation loss (float) and accuracy (float) per epoch.
    """    

    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(val_loader, desc='Validation', position=0, leave=True)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            output_probs = model(inputs)
            loss = criterion(output_probs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(output_probs, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Update the progress bar
            progress_bar.set_postfix({'loss': total_loss / len(val_loader),
                                      'accuracy': 100. * total_correct / total_samples})

    return total_loss / len(val_loader), 100. * total_correct / total_samples


def main():

    """
    Run the training pipeline, including data loading, model training, and model evaluation.
    Generate a confusion matrix and save the model to disk.

    Raises:
        ValueError: If the model type is not supported.
    """

    # Argument parser
    parser = argparse.ArgumentParser(description="Train a VGG model on APTOS dataset.")
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('-m', '--model', type=str, default='vgg', help='Model type (default: vgg).')
    parser.add_argument('--name', type=str, default='default_name', help='Custom name for the model.')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset_path = "./data/aptos2019-blindness-detection/train_images"
    train_loader, val_loader, test_loader = get_dataloaders(dataset_path)

    # Initialize model
    if args.model == 'vgg':
        model = VGGAptos(mode="training")
    else:
        raise ValueError(f"Model type '{args.model}' not supported.")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total parameters.")
    print(f"{trainable_params:,} training parameters.")

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    early_stopper = EarlyStopper(patience=5, min_delta=0.05, verbose=True)

    print('Start training')
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f'Training - Epoch {epoch}, Loss: {train_loss}, Accuracy: {train_acc}')
        print(f'Validation - Epoch {epoch}, Loss: {val_loss}, Accuracy: {val_acc}')
        # early stopping
        if early_stopper(val_loss):
            break
    torch.save(model.state_dict(), args.name+".pth")
    print('Training done')

    # plot confusion matrix
    plot_confusion_matrix(model, test_loader, device, args.name)

if __name__ == '__main__':
    main()
