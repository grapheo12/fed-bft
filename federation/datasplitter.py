import torch.utils.data
from torchvision import datasets, transforms
import os
import pickle
import shutil

def splitData(n=2):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)

    Ntrain = len(dataset1) // n
    Ntest = len(dataset2) // n

    train_lens = [Ntrain for _ in range(n)]
    train_lens[-1] = len(dataset1) - Ntrain * (n - 1)

    test_lens = [Ntest for _ in range(n)]
    test_lens[-1] = len(dataset2) - Ntest * (n - 1)

    train_sets = torch.utils.data.random_split(dataset1, train_lens)
    test_sets = torch.utils.data.random_split(dataset2, test_lens)

    shutil.rmtree("data/processed", ignore_errors=True)
    os.mkdir("data/processed")

    with open("data/processed/train.pkl", "wb") as f:
        pickle.dump(train_sets, f)

    with open("data/processed/test.pkl", "wb") as f:
        pickle.dump(test_sets, f)








