from torchvision import datasets, transforms

def getMNIST():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.EMNIST('data', split="digits", train=True, download=True,
                       transform=transform)
    dataset2 = datasets.EMNIST('data', split="digits", train=False,
                       transform=transform)