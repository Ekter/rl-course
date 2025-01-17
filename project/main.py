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


# game()

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
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model.random_weights()

print(model)

print(model.forward(torch.rand(1, 4).to(device)))

model2 = NeuralNetwork().to(device)
model.random_weights(model2.linear_relu_stack[0], 0.1)

print(model2.forward(torch.rand(1, 4).to(device)))

models = [NeuralNetwork().to(device) for _ in range(10)]

for model in models:
    model.random_weights()
    model.to(device)

scores = []
for model in models:# parallelise this later
    score = launch_game(model)
    print(score)
    scores.append((score,model))

scores.sort(key=lambda x: x[0])

new_models = []

for score, model in scores:
    