# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import Net, PrunableLinear
from utils import sparsity_loss, calculate_sparsity


def train(lambda_val):
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5):
        for images, labels in trainloader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            s_loss = sparsity_loss(model)
            total_loss = loss + lambda_val * s_loss

            total_loss.backward()
            optimizer.step()

    # Evaluation
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity(model)



    gates_all = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            gates_all.extend(gates.flatten())

    plt.hist(gates_all, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
 
    plt.savefig("gate_distribution.png")
    plt.show()




    return accuracy, sparsity


if __name__ == "__main__":
    lambdas = [0.001, 0.01, 0.1]

    for l in lambdas:
        acc, sp = train(l)
        print(f"Lambda: {l} | Accuracy: {acc:.2f}% | Sparsity: {sp:.2f}%")

    