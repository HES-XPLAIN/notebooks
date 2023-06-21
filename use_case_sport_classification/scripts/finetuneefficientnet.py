"""
Fine-tuned EfficientNet Model

This script defines a PyTorch module that implements a fine-tuned version of the EfficientNet model.
The model is initialized with the EfficientNet-B3 architecture and allows for customization of the
number of output classes. The weights of the model can be optionally loaded from pre-trained weights
on the ImageNet dataset. The last fully connected layer and a selected convolutional block are made
trainable while freezing the rest of the model's parameters. This script serves as a template for
fine-tuning EfficientNet models for various image classification tasks.

"""

import torch.nn as nn
import torchvision.models as models

class FineTunedEfficientNet(nn.Module):
    def __init__(self, num_classes, weights='IMAGENET1K_V1'):
        super(FineTunedEfficientNet, self).__init__()

        # Load the EfficientNet-B3 model
        self.model = models.efficientnet_b3(weights=weights)

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
        return self.model(x)
