import os
from fl_client import FLClient


def dataloader():
    import torch
    import torchvision as tv
    transform = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
    ])
    dataset = tv.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)


if __name__ == "__main__":
    client = FLClient('http://localhost:5000')
    data = dataloader()
    for epoch in range(200):
        print(f"epoch {epoch:03d}")
        client.run(data)
