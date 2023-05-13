from torch.autograd import Function
import torch
import fast_sligru_cpp 
import time 


relu = torch.nn.ReLU()

def ligru_cell(wx, u, ht):
    hiddens = []
    for t in range(x.shape[1]):
        gates = wx[:, t] + ht @ u.T
        at, zt = gates.chunk(2, 1)
        zt = torch.sigmoid(zt)
        hcand = relu(at)
        ht = zt * ht + (1 - zt) * hcand 
        hiddens.append(ht)
    
    # Stacking hidden states
    h = torch.stack(hiddens, dim=1)
    return h


def warmup(fct, *kargs, n_iters=2):
    """Warmup function for JiT."""
    for _ in range(n_iters):
        _ = fct(*kargs)

def benchmark(fct, *kargs, n_iters=5):
    """Evaluates an input function over n iterations."""
    avg_time = 0

    torch.cuda.synchronize()
    for _ in range(n_iters):

        torch.cuda.synchronize()
        time1 = time.time()
        _ = fct(*kargs)
        torch.cuda.synchronize()
        avg_time += time.time() - time1

    return avg_time / n_iters    


if __name__ == "__main__":
    batch_size, seq_length, hidden_size, input_size = 4, 10, 4, 10

    x = torch.randn(batch_size, seq_length, input_size, device="cuda")
    w = torch.randn(hidden_size * 2, input_size).to("cuda")
    u = torch.randn(hidden_size * 2, hidden_size).to("cuda")
    ht = torch.zeros(batch_size, hidden_size, device="cuda")
    drop_mask = torch.randn((batch_size, hidden_size), device="cuda", requires_grad=False, dtype=torch.double)
    normalized_shape = u.size(0)
    eps = 1e-5

    wx = x @ w.T 


    fast_sligru_cpp.ligru_forward(wx[:, 0], ht, u, drop_mask, normalized_shape, eps)


    #warmup(ligru_cell, wx, u, ht, n_iters=10)
    #print(benchmark(ligru_cell, wx, u, ht, n_iters=10))