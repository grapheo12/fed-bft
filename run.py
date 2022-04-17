from sys import argv
from model.dataloader import getMNIST
from federation.datasplitter import splitData

if __name__ == "__main__":
    if argv[1] == "getdata":
        getMNIST()
    if argv[1] == "splitdata":
        splitData()
        
