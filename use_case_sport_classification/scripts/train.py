import torch
import torch.nn as nn
import argparse

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from .models import FineTunedEfficientNet
from .helpers import *


def train(model, train_loader, optimizer, criterion):
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
    :return: The average loss and accuracy for the training epoch.
    :rtype: float, float
    """
    model.train()
    print("Training")
    train_loss = 0.0
    train_correct = 0.0
    counter = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels, _ = data
        image = image.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(image)

        # compute loss
        loss = criterion(outputs, labels.long())
        train_loss += loss.item()

        # compute accuracy
        _, preds = torch.max(outputs.data, 1)
        train_correct += (preds == labels).sum().item()

        # backprop
        loss.backward()

        # update the weights
        optimizer.step()

    epoch_loss = train_loss / counter
    epoch_acc = 100. * (train_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """
    Performs validation on a model using a validation dataset for 1 epoch.

    :param model: The model to validate.
    :type model: torch.nn.Module
    :param val_loader: The data loader for the validation dataset.
    :type val_loader: torch.utils.data.DataLoader
    :param criterion: The loss criterion to calculate the loss.
    :type criterion: torch.nn.Module
    :return: The average loss and accuracy for the validation epoch.
    :rtype: float, float
    """
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data, _ in tqdm(enumerate(val_loader), total=len(val_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.float().to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels.long())
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs', type=int, default=20,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '-lr', '--learning-rate', type=float,
        dest='learning_rate', default=0.0001,
        help='Learning rate for training the model'
    )

    parser.add_argument('--name', type=str, default='default_name', help='Custom name')
    args = vars(parser.parse_args())

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    model_name = args['name']


    train_loader, val_loader, _ = get_dataloaders()
    home_path = "/home/arthur.babey/workspace/hes-xplain-arthur/use_case_sport_classification/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FineTunedEfficientNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_model = None
    best_val_acc = -float('inf')
    early_stop_counter = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        print(f"Training loss: {train_loss:.3f}, training acc: {train_acc:.3f}")
        print(f"Validation loss: {val_loss:.3f}, validation acc: {val_acc:.3f}")
        print('-' * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= 6:
            print("early stop")
            break
        print(f"Early counter = {early_stop_counter}")

    # save the trained model weights
    save_model(epochs, model, optimizer, criterion, model_name)
    # save the loss and accuracy plots
    save_plots(train_accs, val_accs, train_losses, val_losses, model_name)