import torch
import torch.nn as nn
import torchvision.models as models

class FineTunedVGG(nn.Module):
    def __init__(self, trunc=7):
        super(FineTunedVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)  # Load pre-trained VGG16 model

        vgg.classifier[-1] = nn.Linear(4096, 100)  # Modify the last fully connected layer
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.trunc = trunc

        # Freeze all layers
        for param in vgg.parameters():
            param.requires_grad = False

        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[:self.trunc](x)  # 3 or 6 (after relu dropout of fc1 or fc2)

        return x