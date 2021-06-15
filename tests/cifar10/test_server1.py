import os
import fl_common as fl
from fl_server import create_app

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value

fl_module = fl.Module()


@fl_module.init
def init(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.flatten = nn.Flatten()
    self.criterion = nn.CrossEntropyLoss()
    self.to('cuda')


@fl_module.infer
def infer(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = self.flatten(x)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x


@fl_module.server_setup
def server_setup(self):
    configure("runs/server1")


@fl_module.on_training_start
def on_training_start(self):
    import torch
    self.to('cuda')
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-4)


@fl_module.training_step
def training_step(self, dataloader):
    self.train()
    for inputs, labels in dataloader:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    return self


@fl_module.aggregation_step
def aggregation_step(self, results):
    with torch.no_grad():
        for a, b in zip(self.parameters(), results[0].parameters()):
            a.copy_(b)


@fl_module.on_aggregation_end
def on_aggregation_end(self):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=os.cpu_count())

    self.eval()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to('cuda')
            labels = labels.to('cuda')

            outputs = self(images)
            loss += self.criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    log_value('valid/loss', loss / len(testloader), step=self._round)
    log_value('valid/acc', correct / total, step=self._round)


if __name__ == "__main__":
    app = create_app(
        fl_module=fl_module,
        n_requests=1,
        instance_path=os.path.join(os.getcwd(), 'instance'))
    app.run()
