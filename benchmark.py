from torch.autograd import Function
import torch
import fast_sligru_cpp 
import time 
from fast_sligru.sligru import SLiGRUCell 

relu = torch.nn.ReLU()
batch_size, seq_length, hidden_size, input_size = 16, 100, 512, 1024
ln = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
double = False 
half = False

@torch.jit.script
def ligru_cell(wx, u, ht, drop_mask):
    hiddens = []

    for t in range(wx.shape[1]):
        gates = wx[:, t] + ln(ht @ u.T)
        at, zt = gates.chunk(2, 1)
        zt = torch.sigmoid(zt)
        hcand = relu(at) * drop_mask
        ht = zt * ht + (1 - zt) * hcand 
        hiddens.append(ht)
    
    # Stacking hidden states
    h = torch.stack(hiddens, dim=1)
    return h

def cuda_ligru_cell(normalized_shape, eps, drop_mask, wx, u, ht):
    hiddens = []

    for t in range(wx.shape[1]):
        ht = fast_sligru_cpp.forward(
            wx[:, t], 
            ht, 
            u,
            drop_mask,
            normalized_shape,
            eps,
            True
        )[0]

        hiddens.append(ht)
    
    # Stacking hidden states
    h = torch.stack(hiddens, dim=1)
    return h



def warmup(fct, *kargs, n_iters=2):
    """Warmup function for JiT."""
    for _ in range(n_iters):
        out = fct(*kargs)
        out.sum().backward(retain_graph=True)

def benchmark(fct, *kargs, n_iters=5):
    """Evaluates an input function over n iterations."""
    avg_time = 0

    torch.cuda.synchronize()
    for _ in range(n_iters):

        torch.cuda.synchronize()
        time1 = time.time()
        out = fct(*kargs)
        out.sum().backward(retain_graph=True)
        torch.cuda.synchronize()
        avg_time += time.time() - time1

    return avg_time / n_iters    


if __name__ == "__main__":

    if double:
        x = torch.randn(batch_size, seq_length, input_size, device="cuda", dtype=torch.float64)
        w = torch.randn(hidden_size * 2, input_size, requires_grad=True, dtype=torch.float64).to("cuda")
        u = torch.randn(hidden_size * 2, hidden_size, requires_grad=True, dtype=torch.float64).to("cuda")
        ht = torch.zeros(batch_size, hidden_size, device="cuda", dtype=torch.float64)
        drop_mask = torch.randn((batch_size, hidden_size), device="cuda", requires_grad=False, dtype=torch.float64)
        normalized_shape = u.size(0)
        eps = 1e-5
    elif half == True : 
        x = torch.randn(batch_size, seq_length, input_size, device="cuda").half()
        w = torch.randn(hidden_size * 2, input_size, requires_grad=True).to("cuda").half()
        u = torch.randn(hidden_size * 2, hidden_size, requires_grad=True).to("cuda").half()
        ht = torch.zeros(batch_size, hidden_size, device="cuda").half()
        drop_mask = torch.randn((batch_size, hidden_size), device="cuda", requires_grad=False).half()
        normalized_shape = u.size(0)
        eps = 1e-5
    else: 
        x = torch.randn(batch_size, seq_length, input_size, device="cuda")
        w = torch.randn(hidden_size * 2, input_size, requires_grad=True).to("cuda")
        u = torch.randn(hidden_size * 2, hidden_size, requires_grad=True).to("cuda")
        ht = torch.zeros(batch_size, hidden_size, device="cuda")
        drop_mask = torch.randn((batch_size, hidden_size), device="cuda", requires_grad=False)
        normalized_shape = u.size(0)
        eps = 1e-5

    wx = x @ w.T 
    
    if double:
        print(torch.autograd.gradcheck(SLiGRUCell.apply, [wx, ht, u, drop_mask]))
    

    out1 = ligru_cell(wx, u, ht, drop_mask)
    out2 = cuda_ligru_cell(normalized_shape, eps, drop_mask, wx, u, ht)
    assert torch.allclose(out1, out2, atol=10-5)
    
    warmup(ligru_cell, wx, u, ht, drop_mask, n_iters=10)
    print(benchmark(ligru_cell, wx, u, ht, drop_mask, n_iters=20))

        
    warmup(SLiGRUCell.apply, wx, ht, u, drop_mask, n_iters=10)
    print(benchmark(SLiGRUCell.apply, wx, ht, u, drop_mask, n_iters=20))