import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value

devcie = torch.device('cuda')
batch_size = 128
epochs = 200
lr = 0.1
weight_decay = 1e-4
# 73.45


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(devcie)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                           num_workers=os.cpu_count(), pin_memory=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                           num_workers=os.cpu_count(), pin_memory=True)


def train_loop(net, criterion, optimizer, trainloader):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(devcie)
        labels = labels.to(devcie)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)


def valid_loop(net, valid_loader):
    net.eval()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(devcie)
            labels = labels.to(devcie)

            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(valid_loader)
    acc = correct / total
    return loss, acc


if __name__ == "__main__":
    print("#.params:", sum(p.numel() for p in net.parameters()))
    print("#.train_iter:", len(train_loader))
    print("#.valid_iter:", len(valid_loader))

    configure("runs/reference")

    for epoch in range(epochs):
        train_loss = train_loop(net, criterion, optimizer, train_loader)
        log_value('train/loss', train_loss, step=epoch)

        valid_loss, valid_acc = valid_loop(net, valid_loader)
        log_value('valid/loss', valid_loss, step=epoch)
        log_value('valid/acc', valid_acc, step=epoch)
