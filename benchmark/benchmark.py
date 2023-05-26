""" This module benchmarks the fast and slow SLiGRU implementations.
"""
import torch
import time 
from fast_sligru import fast_sligru, slow_sligru

def warmup(fct, *kargs, n_iters=2):
    """Warmup function."""
    for _ in range(n_iters):
        out, _ = fct(*kargs)
        out.sum().backward()

def benchmark(fct, *kargs, n_iters=5):
    """Evaluates an input function over n iterations."""
    avg_fwd_time = 0
    avg_bwd_time = 0
    torch.cuda.synchronize()

    for _ in range(n_iters):

        torch.cuda.synchronize()
        time1 = time.time()
        out, _ = fct(*kargs)
        torch.cuda.synchronize()
        avg_fwd_time += time.time() - time1

        out.sum().backward()
        torch.cuda.synchronize()
        avg_bwd_time += time.time() - time1

    return {"avg_fwd_time": avg_fwd_time / n_iters, "avg_bwd_time": avg_bwd_time / n_iters}    

batch_size, hidden_size, input_size, n_layers = 16, 512, 1024, 4
double = False 
half = False

if __name__ == "__main__":

    for seq_length in [100, 500, 1000, 2000, 3000]:
        print("===========================================")
        print(f"seq_length: {seq_length}")

        inp_tensor = torch.rand([batch_size, seq_length, input_size]).to("cuda")
        
        torch.manual_seed(42)
        net1 = fast_sligru.SLiGRU(input_shape=inp_tensor.shape, hidden_size=hidden_size, num_layers=n_layers).to("cuda")
        
        torch.manual_seed(42)
        net2 = slow_sligru.SLiGRU(input_shape=inp_tensor.shape, hidden_size=hidden_size, num_layers=n_layers).to("cuda")

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

        warmup(net1, inp_tensor, n_iters=3)
        print("fast_sligru = ", benchmark(net1, inp_tensor, n_iters=10))

        warmup(net2, inp_tensor, n_iters=3)
        print("slow_sligru = ", benchmark(net2, inp_tensor, n_iters=10))

