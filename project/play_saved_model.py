import torch
from torch import nn
from tqdm import tqdm
import subprocess
# import random
import threading
import time
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")
print("\n\n\n\n\n\n\nfeur")


# def launch_game(model):
#     with open("action.txt","w",encoding="utf-8") as file:
#         file.write("f")
#     subprocess.Popen(["python","shadowgame.py"])
#     print("mpolujyhtfredzs")
#     with open("action.txt","w",encoding="utf-8") as file:
#         file.write("f" if random.random() < 0.5 else "j")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(
                7,
                4,
            ),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.Softmax(dim=0),
        )

    def random_weights(self, base=None, std=1):
        if base is None:
            base = self.linear_relu_stack
        for layer in base:
            if isinstance(layer, nn.Linear):
                layer.weight.data = (
                    layer.weight.data.clone() + torch.randn_like(layer.weight.data) * std
                )
                layer.bias.data = (
                    layer.bias.data.clone() + torch.randn_like(layer.bias.data) * std
                )
                print(layer.weight.data)
                print(layer.bias.data)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def launch_game(model, i):
    if os.path.exists(f"data/action{i}.txt" == False):
        with open(f"data/action{i}.txt", "w", encoding="utf-8") as file:
            file.write("f")
    if os.path.exists(f"data/discore{i}.txt" == False):
        with open(f"data/discore{i}.txt", "w", encoding="utf-8") as file:
            file.write("1, 1, 1, 1, 1, 1")
    subprocess.Popen(["python", "shadowgame.py", str(i)])
    while True:
        time.sleep(1 / 30)
        with open(f"data/discore{i}.txt", "r", encoding="utf-8") as file:
            reading = file.read()
        # print(reading)
        reading = reading.replace("(", "").replace(")", "")
        data = reading.split(", ")
        while len(data) < 9:
            with open(f"data/discore{i}.txt", "r", encoding="utf-8") as file:
                reading = file.read()
            reading = reading.replace("(", "").replace(")", "")
            data = reading.split(", ")
            # print("len pb ===========================================")
        if data[8] == "True":
            return int(data[7])
        model_input = model.forward(
            torch.tensor(
                [float(x) for x in data[0:7]],
                dtype=torch.float,
            ).to(device)
        )
        with open(f"data/action{i}.txt", "w", encoding="utf-8") as file:
            match torch.argmax(model_input):
                case 0:
                    file.write("j")
                case 1:
                    file.write("f")
                case 2:
                    file.write("r")


def play(model, times):
    model = torch.load(model).to(device)

    for _ in tqdm(range(times)):
        score = launch_game(model, 0)
        print(score)


if __name__ == "__main__":
    play("models/model1737406801.8995628_3837_3_9.pt", times=2)
