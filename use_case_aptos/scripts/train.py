import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import VGGAptos
from helpers import get_dataloaders, save_model, save_plots, plot_confusion_matrix


def train(model, train_loader, optimizer, criterion, device):
    """
    Performs training on a model using a training dataset.

    :param model: The model to train.
    :type model: torch.nn.Module
    :param train_loader: The data loader for the training dataset.
    :type train_loader: torch.utils.data.DataLoader
    :param optimizer: The optimizer for updating the model's weights.
    :type optimizer: torch.optim.Optimizer
    :param criterion: The loss criterion to calculate the loss.
    :type criterion: torch.nn.Module
    :param device: The device to perform training on.
    :type device: torch.device
    :return: The average loss and accuracy for the training epoch.
    :rtype: float, float
    """
    model.train()
    train_loss = 0.0
    train_correct = 0.0

    for image, labels in tqdm(train_loader):
        image, labels = image.to(device), labels.float().to(device)
        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, labels.long())
        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()

    epoch_loss = train_loss / len(train_loader)
    epoch_acc = 100. * (train_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Performs validation on a model using a validation dataset for 1 epoch.

    :param model: The model to validate.
    :type model: torch.nn.Module
    :param val_loader: The data loader for the validation dataset.
    :type val_loader: torch.utils.data.DataLoader
    :param criterion: The loss criterion to calculate the loss.
    :type criterion: torch.nn.Module
    :param device: The device to perform validation on.
    :type device: torch.device
    :return: The average loss and accuracy for the validation epoch.
    :rtype: float, float
    """
    model.eval()
    valid_loss = 0.0
    valid_correct = 0

    with torch.no_grad():
        for image, labels in tqdm(val_loader):
            image, labels = image.to(device), labels.float().to(device)

            outputs = model(image)
            loss = criterion(outputs, labels.long())
            valid_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            valid_correct += (preds == labels).sum().item()

    epoch_loss = valid_loss / len(val_loader)
    epoch_acc = 100. * (valid_correct / len(val_loader.dataset))
    return epoch_loss, epoch_acc


def main():
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
    # TODO: remove hard code
    dataset_path = "/home/arthur.babey/workspace/project/aptos/data/aptos2019-blindness-detection_augmented/train_images"
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

    # Training loop
    best_val_acc = -float('inf')
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {args.epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_acc)

        print(f"Training loss: {train_loss:.3f}, Training acc: {train_acc:.3f}")
        print(f"Validation loss: {val_loss:.3f}, Validation acc: {val_acc:.3f}")
        print('-' * 50)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 6:
            print("Early stopping triggered.")
            break

        print(f"Early stopping counter: {early_stop_counter}")

    # Save the best model, plots, and confusion matrix
    save_model(args.epochs, model, optimizer, criterion, args.name)
    save_plots(train_accs, val_accs, train_losses, val_losses, args.name)
    plot_confusion_matrix(model, test_loader, device, args.name)


if __name__ == '__main__':
    main()
