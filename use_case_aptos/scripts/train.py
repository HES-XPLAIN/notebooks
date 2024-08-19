import torch
import torch.nn as nn
from torchvision import models, transforms

from .model import VGGAptos
import argparse

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
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            counter += 1
            image, labels, _ = data
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
        '-e', '--epochs', type=int, default=50,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '-lr', '--learning-rate', type=float,
        dest='learning_rate', default=0.0001,
        help='Learning rate for training the model'
    )

    parser.add_argument('-m', '--model', type=str, default='vgg', help='model type: vgg or efficientnet')
    parser.add_argument('--name', type=str, default='default_name', help='Custom name')
    args = vars(parser.parse_args())

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    model_type = args['model']
    model_name = args['name']

    train_loader, val_loader, _ = get_dataloaders() # TODO: Implement get_dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'vgg':
        model = VGGAptos()
    else:
        raise ValueError('Model type not supported')
    