from sys import argv
import os
import shutil
from model.dataloader import getMNIST
from federation.datasplitter import splitData
from federation.master import createModel, fedTrain, collectModel
from federation.worker import run_app

if __name__ == "__main__":
    if argv[1] == "setup":
        if not os.path.exists("data"):
            os.mkdir("data")

        shutil.rmtree("data/processed", ignore_errors=True)
        shutil.rmtree("outputs", ignore_errors=True)
        os.mkdir("data/processed")
        os.mkdir("outputs")

    if argv[1] == "getdata":
        getMNIST()

    if argv[1] == "splitdata":
        if len(argv) == 3:
            splitData(int(argv[2]))
        else:
            splitData()

    if argv[1] == "createmodel":
        createModel()

    if argv[1] == "train":
        C = int(argv[2])
        addrs = argv[3:]
        fedTrain(addrs, C)

    if argv[1] == "worker":
        idx = int(argv[2])
        port = int(argv[3])
        byz = False
        if len(argv) == 5 and argv[4] == "byz":
            byz = True

        run_app(idx, port, byzantine=byz)

    if argv[1] == "collect":
        sizes = {int(x.split(":")[0]): int(x.split(":")[1]) for x in argv[2:]}
        collectModel(sizes)

    if argv[1] == "clean":
        for f in os.listdir("outputs"):
            if f != "main.pt":
                os.remove("outputs/" + f)


        
