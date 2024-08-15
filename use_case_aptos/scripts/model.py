import torch
import torch.nn as nn
import torchvision.models as models

class VGGAptos(nn.Module):
    def __init__(self, num_classes=5):
        super(VGGAptos, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, num_classes)
        
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x