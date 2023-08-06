import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

from fast_sligru import fast_sligru, fast_ligru, slow_gru, slow_lstm

class Model(nn.Module):
    def __init__(self, rnn, hidden_size):
        super().__init__()
        self.rnn = rnn
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input, hx=None):
        if hx is None:
            output, _ = self.rnn(input)
        else:
            output, _ = self.rnn(input, hx)

        output = self.output_layer(output[:, -1, :])
        return output

def generate_add_example(seq_length):
    b1 = random.randint(0, seq_length//2 - 1)
    b2 = random.randint(seq_length//2, seq_length - 1)
    
    mask = [0.] * seq_length
    mask[b1] = 1.
    mask[b2] = 1.

    x = [(random.uniform(0, 1), marker) for marker in mask]
    y = x[b1][0] + x[b2][0]
    
    return x, y

def generate_batch(seq_length, batch_size):
    
    n_elems = 2
    x = np.empty((batch_size, seq_length, n_elems))
    y = np.empty((batch_size, 1))

    for i in range(batch_size):
        sample, ground_truth = generate_add_example(seq_length=seq_length)
        x[i, :, :] = sample 
        y[i, 0] = ground_truth
    return x, y

if __name__ == "__main__":
    batch_size = 128
    seq_length = 2_000
    hidden_size = 256
    num_layer = 1
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    learning_rate = 0.001 # 0.001
    epochs = 2_000  
    cuda_graph = True
    jit_compile = False

    rnn = slow_lstm.LSTM( 
        input_shape=(batch_size, seq_length, 2),
        hidden_size=hidden_size,
        num_layers=num_layer,
        dropout=dropout,
        bidirectional=False,
        nonlinearity="tanh",
    )

    if jit_compile:
        rnn = torch.jit.script(rnn)

    net = Model(rnn, hidden_size=hidden_size).to(device).to(dtype=dtype)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, capturable=True)
    mse_loss_fn = nn.MSELoss()

    net.train()
    if cuda_graph:
        x, y = generate_batch(
            seq_length=seq_length,
            batch_size=batch_size,
        )
        
        x_static = torch.tensor(x, device=device, requires_grad=False).to(dtype=dtype)
        y_static = torch.tensor(y, device=device, requires_grad=False).to(dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for i in range(5):
                optimizer.zero_grad(set_to_none=True)
                y_pred = net(x_static)
                loss = mse_loss_fn(y_pred, y_static)
                loss.backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()

        # need to be global to be accessible
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            y_pred = net(x_static)
            loss = mse_loss_fn(y_pred, y_static)
            loss.backward()
            optimizer.step()

        for epoch in range(epochs+1):
            x, y = generate_batch(
                seq_length=seq_length,
                batch_size=batch_size,
            )
            x = torch.tensor(x, device=device, requires_grad=False).to(dtype=dtype)
            y = torch.tensor(y, device=device, requires_grad=False).to(dtype=dtype)
        
            x_static.copy_(x)
            y_static.copy_(y)
            g.replay()

            with torch.no_grad():
                if epoch % 1 == 0:
                    print("-"*50)
                    print(f"Epoch: {epoch} Loss: {loss.item()}")
                    uh, uz = net.rnn.rnn[0].u.weight.chunk(2, dim=1)
                    uh_norm = torch.linalg.matrix_norm(uh, ord=2) 
                    uz_norm = torch.linalg.matrix_norm(uz, ord=2)
                    print(f"uh_norm_2: {uh_norm.item()} uz_norm_2: {uz_norm.item()}")
                    uh_grad, uz_grad = net.rnn.rnn[0].u.weight.grad.chunk(2, dim=1)
                    uh_grad_norm = torch.linalg.matrix_norm(uh_grad, ord=2)
                    uz_grad_norm = torch.linalg.matrix_norm(uz_grad, ord=2)
                    print(f"uh_norm_2_grad: {uh_grad_norm.item()} uz_norm_2_grad: {uz_grad_norm.item()}")
                    print("-"*50)
    else:
        for epoch in range(epochs+1):
            x, y = generate_batch(
                seq_length=seq_length,
                batch_size=batch_size,
            )
            x = torch.tensor(x, device=device, requires_grad=False).to(dtype=dtype)
            y = torch.tensor(y, device=device, requires_grad=False).to(dtype=dtype)

            optimizer.zero_grad(set_to_none=True)
            y_pred = net(x)
            loss = mse_loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch % 1 == 0:
                    print("-"*50)
                    print(f"Epoch: {epoch} Loss: {loss.item()}")
                    uh, uz = net.rnn.rnn[0].u.weight.chunk(2, dim=1)
                    uh_norm = torch.linalg.matrix_norm(uh, ord=2) 
                    uz_norm = torch.linalg.matrix_norm(uz, ord=2)
                    print(f"uh_norm_2: {uh_norm.item()} uz_norm_2: {uz_norm.item()}")
                    uh_grad, uz_grad = net.rnn.rnn[0].u.weight.grad.chunk(2, dim=1)
                    uh_grad_norm = torch.linalg.matrix_norm(uh_grad, ord=2)
                    uz_grad_norm = torch.linalg.matrix_norm(uz_grad, ord=2)
                    print(f"uh_norm_2_grad: {uh_grad_norm.item()} uz_norm_2_grad: {uz_grad_norm.item()}")
                    print("-"*50)