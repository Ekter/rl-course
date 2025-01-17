
from shadowgame import *
from ursina import *
import torch
from torch import nn
from tqdm import tqdm
import subprocess

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
print("\n\n\n\n\n\n\nfeur")



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

def launch_game():
    with open("action.txt","w",encoding="utf-8") as file:
        file.write("f")
    subprocess.Popen(["python","shadowgame.py"])
    print("mpolujyhtfredzs")


if __name__ == "__main__":
    launch_game()