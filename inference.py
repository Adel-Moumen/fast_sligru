import torch
from fast_ligru_pytorch.stabilised_ligru import SLiGRU

batch, time, feats = 10, 10, 10
hidden_size, num_layer, dropout = 512, 4, 0.1
nonlinearity = "relu" # works also with sine, leakyrelu and tanh

device = "cuda" # it works with CPU too

inp_tensor = torch.randn((batch, time, feats), requires_grad=False).to(device)
net = SLiGRU( # or LiGRU
    input_shape=inp_tensor.shape,
    hidden_size=hidden_size,
    num_layers=num_layer,
    dropout=dropout,
    nonlinearity=nonlinearity,
).to(device)

# forward
out_tensor, _ = net(inp_tensor)
print(out_tensor)

# backward
out_tensor.sum().backward()