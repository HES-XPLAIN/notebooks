import torch
import os
import argparse

from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from .helpers import get_dataloaders
from .finetunedefficientnet import FineTunedEfficientNet

def evaluate(model_path, test_loader):
    """
    Evaluates the performance of a trained model on a test dataset.

    :param model_path: The path to the trained model's checkpoint.
    :type model_path: str
    :param test_loader: The data loader for the test dataset.
    :type test_loader: torch.utils.data.DataLoader
    :return: The accuracy, F1 score, and confusion matrix of the model.
    :rtype: float, float, numpy.ndarray
    """
    # loading model
    model = FineTunedEfficientNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The device is: {device}")
    checkpoint = torch.load(model_path, map_location=device)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    correct, total, f1 = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        model.eval()
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)  # move the labels to the device

            # forward pass
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            # update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro') * labels.size(0)
            # append true and predicted labels for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    F1s = 100 * f1 / total
    accuracy = 100 * correct / total

    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Test F1 score: {F1s:.2f}%")

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, F1s, cm


if __name__ == '__main__':
    home_path = "/home/arthur.babey/workspace/hes-xplain-arthur/use_case_sport_classification"
    saved_model_path = os.path.join(home_path, "models", "saved_models")

    parser = argparse.ArgumentParser(description="Evaluating script")
    parser.add_argument('--model_name', type=str, default='model', help='name of the model you want to evaluate')
    args = parser.parse_args()

    modelname = args.model_name
    model_path = os.path.join(saved_model_path, modelname)

    _, _, test_loader = get_dataloaders()
    _, _, _ = evaluate(model_path=model_path, test_loader=test_loader)