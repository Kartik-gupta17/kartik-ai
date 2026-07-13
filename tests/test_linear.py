import torch

from ai.nn.linear import Linear


layer = Linear(
    in_features=4,
    out_features=3,
)

x = torch.randn(2, 4)

y = layer(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)

print(y)