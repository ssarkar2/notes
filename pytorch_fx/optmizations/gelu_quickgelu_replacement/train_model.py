import torch
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os

from replace_gelu import replace_gelu_approximate_with_tanh, replace_gelu_with_relu, replace_relu_with_quickgelu
import torch.fx as fx
import time


# wrap in main, else num_workers>0 hangs on macOS
def main():
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data transforms and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


    # Data preparation
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    def get_test_loader():
        test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        return test_loader

    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = get_test_loader()

    # Simple vision model with GELU
    class SimpleVisionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(64, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    checkpoint_path = "./simple_vision_net_checkpoint.pth"
    model = SimpleVisionNet().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found, training from scratch.")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(5):
            model.train()
            start_time = time.time()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            end_time = time.time()
            print(f"Epoch {epoch+1} complete. Time taken: {end_time - start_time:.2f} seconds")

        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def eval(model, test_loader, device):
        # Evaluation
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        end_time = time.time()
        print(f"Evaluation complete. device={device}, Time taken: {end_time - start_time:.2f} seconds")
        print(f"Test accuracy: {correct / total:.2%}")

    print('Evaluating original model:')
    eval(model, get_test_loader(), device)
    eval(model, get_test_loader(), 'cpu')

    model = fx.symbolic_trace(model)
    print('Evaluating on traced model:')
    eval(model, get_test_loader(), device)
    eval(model, get_test_loader(), 'cpu')

    transformations = [
            replace_gelu_approximate_with_tanh,
            replace_gelu_with_relu,
            replace_relu_with_quickgelu,
        ]

    for transform in transformations:
        model = transform(model)
        print(f'transform = {transform.__name__}:')
        eval(model, get_test_loader(), device)
        eval(model, get_test_loader(), 'cpu')


    '''
    Evaluating original model:
    Evaluation complete. device=mps, Time taken: 1.44 seconds
    Test accuracy: 54.36%
    Evaluation complete. device=cpu, Time taken: 5.55 seconds
    Test accuracy: 54.36%
    Evaluating on traced model:
    Evaluation complete. device=mps, Time taken: 1.65 seconds
    Test accuracy: 54.36%
    Evaluation complete. device=cpu, Time taken: 5.82 seconds
    Test accuracy: 54.36%
    transform = replace_gelu_approximate_with_tanh:
    Evaluation complete. device=mps, Time taken: 1.31 seconds
    Test accuracy: 54.33%
    Evaluation complete. device=cpu, Time taken: 6.26 seconds
    Test accuracy: 54.33%
    transform = replace_gelu_with_relu:
    Evaluation complete. device=mps, Time taken: 1.17 seconds
    Test accuracy: 27.59%
    Evaluation complete. device=cpu, Time taken: 5.25 seconds
    Test accuracy: 27.59%
    transform = replace_relu_with_quickgelu:
    Evaluation complete. device=mps, Time taken: 1.61 seconds
    Test accuracy: 54.22%
    Evaluation complete. device=cpu, Time taken: 5.67 seconds
    Test accuracy: 54.22%
    '''

if __name__ == "__main__":
    main()