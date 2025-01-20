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
            nn.Linear(4, 3, ),
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
    if  os.path.exists(f"data/action{i}.txt" == False):
        with open(f"data/action{i}.txt","w",encoding="utf-8") as file:
            file.write("f")
    if os.path.exists(f"data/discore{i}.txt" == False):
        with open(f"data/discore{i}.txt", "w", encoding="utf-8") as file :
            file.write("1, 1, 1, 1, 1, 1")
    subprocess.Popen(["python","shadowgame.py", str(i)])
    while True:
        time.sleep(1/30)
        with open(f"data/discore{i}.txt","r",encoding="utf-8") as file:
            reading = file.read()
        # print(reading)
        reading = reading.replace("(","").replace(")","")
        data = reading.split(", ")
        while len(data) < 6 : 
            with open(f"data/discore{i}.txt","r",encoding="utf-8") as file:
                reading = file.read()
            reading = reading.replace("(","").replace(")","")
            data = reading.split(", ")
            # print("len pb ===========================================")
        if data[4] == "True":
            return int(data[3])
        model_input = model.forward(torch.tensor((int(data[0]),int(data[1]),int(data[2]),int(data[5])),dtype=torch.float).to(device))
        with open(f"data/action{i}.txt","w", encoding="utf-8") as file :
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

class ThreadWithReturnValue(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

def train(epochs):
    models = [NeuralNetwork().to(device) for _ in range(10)]

    for model in models:
        model.random_weights()
        model.to(device)

    for epoch in tqdm(range(epochs)):

        scores = []
        threads = []
        for i, model in enumerate(models):# parallelise this later
            threads.append(ThreadWithReturnValue(target=launch_game, args=(model, i)))
            # threads[i].start()
        for i, model in enumerate(models):
            score = launch_game(model, i)
            # score = threads[i].join()
            print(score)
            scores.append((score,model))

        scores.sort(key=lambda x: x[0])

        new_models = []

        for score, model in scores:# improve model
            new_models.append(model.random_weights(model.linear_relu_stack[0], 1/score))



if __name__ == "__main__":
    train(5)
