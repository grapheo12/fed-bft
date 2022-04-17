import shutil
from time import sleep
from model.nn import Net
import torch.nn as nn
import torch
import requests
import random
import os

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def createModel():
    net = Net()
    net.apply(init_weights)

    shutil.rmtree("outputs/*.pt", ignore_errors=True)
    torch.save(net.state_dict(), "outputs/main.pt")


def fedTrain(nodes, C):
    # Randomly choose C nodes and send requests for training
    n = random.sample(nodes, C)
    for addr in enumerate(n):
        print("Sending request to", addr)
        rq = requests.get("http://" + addr + "/train")
        print(rq.text)


def collectModel(sizes):
    while True:
        print("Checking for availability of worker models...")
        l = os.listdir("outputs")
        cnt = len([a.startswith("worker") for a in l])
        if cnt >= len(sizes):
            break
        sleep(1)

    worker_models = []
    sizemap = []
    for f in os.listdir("outputs"):
        if f.startswith("worker"):
            net = Net()
            net.load_state_dict(torch.load("outputs/" + f))
            worker_models.append(net)
            sizemap.append(sizes[int(f.split("worker")[1].split(".")[0])])
    
    total_size = sum(sizemap)
    weights = [a / total_size for a in sizemap]

    net = Net()
    sd = net.state_dict()

    print("Applying FedAvg...")
    # Federated Averaging Algorithm
    for layer, param in sd.items():
        _param = worker_models[0].state_dict()[layer] * weights[0]
        for i in range(1, len(weights)):
            _param += worker_models[i].state_dict()[layer] * weights[i]

        param.copy_(_param)

    os.remove("outputs/main.pt")
    torch.save(net.state_dict(), "outputs/main.pt")
    print("Model Saved")
            

