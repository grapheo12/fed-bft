import pickle
from sys import stdout
from flask import Flask, jsonify
import torch
import torch.utils.data
import torch.optim as optim
from model.nn import Net, train
from threading import Thread

def train_proc(model, op, data, id, byz=False):
    train(model, "cpu", data, op, byz=byz)
    torch.save(model.state_dict(), "outputs/worker" + str(id) + ".pt")


def run_app(id, port, byzantine=False):
    app = Flask(__name__)

    @app.route("/train")
    def train():
        model = Net()
        model.load_state_dict(torch.load("outputs/main.pt"))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        with open("data/processed/train.pkl", "rb") as f:
            train_sets = pickle.load(f)
        train_loader = torch.utils.data.DataLoader(train_sets[id], batch_size=1000)
        p = Thread(target=train_proc,
            args=(model, optimizer, train_loader, id, byzantine))
        p.start()

        return jsonify({
            "size": len(train_sets[id])
        }), 200

    app.run(host="0.0.0.0", port=port, debug=True)
