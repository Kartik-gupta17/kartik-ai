import torch

from ai.nn import Dropout


drop = Dropout(0.5)

drop.train()

x = torch.ones(5, 5)

print("Training Mode")

print(drop(x))

print()

drop.eval()

print("Evaluation Mode")

print(drop(x))