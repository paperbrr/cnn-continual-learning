import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def get_VGG16(out_features:int):

    VGG16 = torchvision.models.vgg16(weights=None)
    VGG16.classifier[-1] = nn.Linear(in_features=VGG16.classifier[-1].in_features, out_features=out_features, bias=True)
    VGG16.apply(init_weights)

    return VGG16


def get_ResNet18(out_features:int):

    ResNet18 = torchvision.models.resnet18(weights=None)
    ResNet18.fc = torch.nn.Linear(in_features=ResNet18.fc.in_features, out_features=out_features, bias=True)
    ResNet18.apply(init_weights)

    return ResNet18