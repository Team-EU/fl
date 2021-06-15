import os
from fl.client import FLClient
import argparse


def dataloader(classes = None):
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
    if classes:
        sidx = [idx for idx, t in enumerate(dataset.targets) if t in classes]
        dataset = torch.utils.data.Subset(dataset, sidx)
    print(len(dataset))
    return torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, nargs='+', help='class labels for non iid tests')
    parser.add_argument('--host', help='server host', default='localhost')
    parser.add_argument('--port', help='server host', default='5000')
    args = parser.parse_args()
    client = FLClient(f'http://{args.host}:{args.port}')
    data = dataloader(args.classes)
    for epoch in range(200):
        client.pull(round=epoch)
        client.run(data)
