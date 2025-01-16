
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

Shadgame = ShadowGame()
input = Shadgame.input
update = Shadgame.update

def game():
    window.fullscreen = True
    window.color = color.white
    camera.orthographic = True
    camera.fov = 10
    Shadgame.app.run()


game()

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

