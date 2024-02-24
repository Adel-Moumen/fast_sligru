""" This module benchmarks the fast and slow SLiGRU implementations.
"""
import torch
import time 
from fast_sligru import fast_ligru, slow_sligru

def warmup(fct, *kargs, n_iters=2):
    """Warmup function."""
    for _ in range(n_iters):
        out = fct(*kargs)
        # out # .sum().backward()

def benchmark(fct, *kargs, n_iters=5):
    """Evaluates an input function over n iterations."""
    avg_fwd_time = 0
    avg_bwd_time = 0
    torch.cuda.synchronize()

    for _ in range(n_iters):

        torch.cuda.synchronize()
        time1 = time.time()
        out = fct(*kargs)
        torch.cuda.synchronize()
        avg_fwd_time += time.time() - time1


    return {"avg_fwd_time": avg_fwd_time / n_iters}    

batch_size, hidden_size, input_size, n_layers = 16, 512, 1024, 4
double = False 
half = False
device = "cuda"

@torch.compile(mode="max-autotune")
def one_layer_loss(rnn_layer):
    uh, uz = rnn_layer.u.weight.chunk(2, dim=1)
    eta = rnn_layer.hh_max / 4 * torch.linalg.matrix_norm(uh, ord=2) + torch.linalg.matrix_norm(uz, ord=2)
    return torch.abs(eta - 1)

def compute_loss(net):
    loss = torch.tensor(0., device=net.rnn[0].u.weight.device)
    for rnn_layer in net.rnn:
        loss += one_layer_loss(rnn_layer)
    return loss / len(net.rnn) 

@torch.jit.script
def jit_one_layer_loss(hh_max, uh, uz):
    eta = hh_max / 4 * torch.linalg.matrix_norm(uh, ord=2) + torch.linalg.matrix_norm(uz, ord=2)
    return torch.abs(eta - 1)

def jit_compute_loss(net):
    loss = torch.tensor(0., device=net.rnn[0].u.weight.device)
    for rnn_layer in net.rnn:
        uh, uz = rnn_layer.u.weight.chunk(2, dim=1)
        loss += jit_one_layer_loss(rnn_layer.hh_max, uh, uz)
    return loss / len(net.rnn)

if __name__ == "__main__":

    torch.manual_seed(42)
    net1 = fast_ligru.LiGRU(input_shape=(batch_size, 1, input_size), hidden_size=hidden_size, num_layers=n_layers, bidirectional=True).to(device)
        
    torch.manual_seed(42)
    net2 = slow_sligru.SLiGRU(input_shape=(batch_size, 1, input_size), hidden_size=hidden_size, num_layers=n_layers).to(device)

    for seq_length in [100, 500, 1000, 2000, 3000]:
        print("===========================================")
        print(f"seq_length: {seq_length}")

        inp_tensor = torch.rand([batch_size, seq_length, input_size]).to(device)

        if double:
            inp_tensor = inp_tensor.double()
            net1 = net1.double()
            net2 = net2.double()
        elif half == True : 
            inp_tensor = inp_tensor.half()
            net1 = net1.half()
            net2 = net2.half()
        else: 
            inp_tensor = inp_tensor.float()
            net1 = net1.float()
            net2 = net2.float()

        net1(inp_tensor)
        warmup(compute_loss, net1, n_iters=3)
        print("no jit = ", benchmark(compute_loss, net1, n_iters=10))

        warmup(jit_compute_loss, net1, n_iters=3)
        print("jit = ", benchmark(jit_compute_loss, net1, n_iters=10))


