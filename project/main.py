from shadowgame import *
from ursina import *
import torch
from torch import nn
from tqdm import tqdm
import subprocess
import random
import threading
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
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
            nn.Linear(3, 3, ),
            # nn.ReLU(),
            # nn.Linear(3, 3),
            nn.Softmax(),
        )

    def random_weights(self, base = None, std = 1):
        if base is None:
            base = self.linear_relu_stack[0]
        for layer in self.linear_relu_stack:
            if isinstance(layer, nn.Linear):
                layer.weight.data = base.weight.data.clone()+torch.randn_like(layer.weight.data)*std
                layer.bias.data = base.bias.data.clone()+torch.randn_like(layer.bias.data)*std
                print(layer.weight.data)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def launch_game(model, i):
    with open(f"action{i}.txt","w",encoding="utf-8") as file:
        file.write("f")
    with open(f"discore{i}.txt", "w", encoding="utf-8") as file :
        file.write("1, 1, 1, 1, 1, 1")
    subprocess.Popen(["python","shadowgame.py", str(i)])
    while True:
        time.sleep(1/30)
        with open(f"discore{i}.txt","r",encoding="utf-8") as file:
            reading = file.read()
        print(reading)
        reading = reading.replace("(","").replace(")","")
        data = reading.split(", ")
        if data[4] == "True":
            return data[3]
        model_input = model.forward(torch.tensor((int(data[0]),int(data[1]),int(data[2])),dtype=torch.float).to(device))
        with open(f"action{i}","w", encoding="utf-8") as file :
            match torch.argmax(model_input) :
                case 0 :
                    file.write("j")
                case 1 :
                    file.write("f")
                case 2 : 
                    file.write("r")



# model = NeuralNetwork().to(device)
# model.random_weights()

# print(model)

# print(model.forward(torch.rand(1, 4).to(device)))

# model2 = NeuralNetwork().to(device)
# model.random_weights(model2.linear_relu_stack[0], 0.1)

# print(model2.forward(torch.rand(1, 4).to(device)))

def train(epochs):
    models = [NeuralNetwork().to(device) for _ in range(10)]

    for model in models:
        model.random_weights()
        model.to(device)

    for epoch in tqdm(range(epochs)):

        scores = []
        for i, model in enumerate(models):# parallelise this later
            score = launch_game(model, i)
            print(score)
            scores.append((score,model))

        scores.sort(key=lambda x: x[0])

        new_models = []

        for score, model in scores:# improve model
            new_models.append(model.random_weights(model, scores[0][1], 0.1))



if __name__ == "__main__":
    train(5)
