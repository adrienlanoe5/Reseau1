import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# lets try this with Classification problem
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

print(resnet18_model)

data_transform = transforms.Compose([
    transforms.Resize(size=(28, 28)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(10),  # Added rotation
    transforms.ToTensor(),

])

train_data = datasets.ImageFolder(root="/kaggle/input/geometric-shapes-mathematics/dataset/train",
                                  transform=data_transform, # a transform for the data
                                  target_transform=None) # a transform for the label/target

test_data = datasets.ImageFolder(root="/kaggle/input/geometric-shapes-mathematics/dataset/test",
                                 transform=data_transform)


validation_data = datasets.ImageFolder(root="/kaggle/input/geometric-shapes-mathematics/dataset/val",
                                 transform=data_transform)
train_data, test_data ,validation_data

train_data[0]