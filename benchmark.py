from torch.autograd import Function
import torch
import fast_sligru_cpp 
import time 
from fast_sligru.sligru import SLiGRU
import fast_sligru
from fast_sligru.sligru_vanilla import LiGRU

relu = torch.nn.ReLU()
batch_size, seq_length, hidden_size, input_size, n_layers = 16, 1_000, 512, 1024, 4
ln = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
double = False 
half = False

def warmup(fct, *kargs, n_iters=2):
    """Warmup function for JiT."""
    for _ in range(n_iters):
        out, _ = fct(*kargs)
        out.sum().backward()

def benchmark(fct, *kargs, n_iters=5):
    """Evaluates an input function over n iterations."""
    avg_time = 0

    torch.cuda.synchronize()
    for _ in range(n_iters):

        torch.cuda.synchronize()
        time1 = time.time()
        out, _ = fct(*kargs)
        out.sum().backward()
        torch.cuda.synchronize()
        avg_time += time.time() - time1

    return avg_time / n_iters    


if __name__ == "__main__":
    inp_tensor = torch.rand([batch_size, seq_length, input_size]).to("cuda")
    torch.manual_seed(42)
    net1 = SLiGRU(input_shape=inp_tensor.shape, hidden_size=hidden_size, num_layers=n_layers).to("cuda")
    torch.manual_seed(42)
    net2 = LiGRU(input_shape=inp_tensor.shape, hidden_size=hidden_size, num_layers=n_layers).to("cuda")


    """
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
    """
    net1 = torch.jit.script(net1)
    net2 = torch.jit.script(net2)

    warmup(net1, inp_tensor, n_iters=3)
    print(benchmark(net1, inp_tensor, n_iters=10))

    warmup(net2, inp_tensor, n_iters=3)
    print(benchmark(net2, inp_tensor, n_iters=10))
