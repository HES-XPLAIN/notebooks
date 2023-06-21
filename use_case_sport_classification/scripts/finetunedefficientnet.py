import torch.nn as nn
import torchvision.models as models


class FineTunedEfficientNet(nn.Module):
    def __init__(self, num_classes=100, weights='IMAGENET1K_V1'):
        """
        Initializes a fine-tuned EfficientNet model.

        :param num_classes: The number of output classes.
        :type num_classes: int
        :param weights: The pre-trained weights to load for the EfficientNet model.
        :type weights: str
        """
        super(FineTunedEfficientNet, self).__init__()

        # Load the EfficientNet-B0 model
        self.model = models.efficientnet_b0(weights=weights)

        # Retrieve the number of input features and dropout probability from the original classifier
        num_features = self.model.classifier[1].in_features
        dropout = self.model.classifier[0].p

        # Replace the original classifier with a new one for the desired number of output classes
        self.model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(num_features, num_classes))

        # Freeze all the parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of the last fully connected layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Unfreeze the parameters of the last convolutional block (-4 refers to the 4th-to-last block)
        for param in self.model.features[-4].parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the fine-tuned EfficientNet model.

        :param x: The input tensor.
        :type x: torch.Tensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self.model(x)
