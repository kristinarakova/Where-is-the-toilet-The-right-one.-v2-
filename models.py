import torch
from torch import nn
from torchvision import models


def set_parameter_requires_grad(model, train_last):
    if train_last:
        for param in model.parameters():
            param.requires_grad = False


def resnext50_32x4d(train_last=True, pretrained=True):
    model = models.resnext50_32x4d(pretrained=pretrained)
    set_parameter_requires_grad(model, train_last)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
    )
    return model


def densenet161(train_last=True, pretrained=True):
    model = models.densenet161(pretrained=pretrained)
    set_parameter_requires_grad(model, train_last)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
    )
    return model


def mobilenet_v2(train_last=True, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    set_parameter_requires_grad(model, train_last)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
    )
    return model


def vgg16(train_last=True, pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    set_parameter_requires_grad(model, train_last)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier._modules["0"].in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
    )
    return model


def make_model(train_last_layers, pretrained, device, model_type, weights=None):
    model = model_type(train_last_layers, pretrained)
    if weights is not None:
        model_state_dict = torch.load(weights)
        model.load_state_dict(model_state_dict)
    model = model.to(device)
    params_to_update = model.parameters()
    if train_last_layers:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    return model, params_to_update
