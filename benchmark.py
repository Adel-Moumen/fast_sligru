from torch.autograd import Function
import torch
import fast_sligru_cpp 
import time 


relu = torch.nn.ReLU()
batch_size, seq_length, hidden_size, input_size = 16, 2_000, 1024, 1024
ln = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
double = False
half = True

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
            eps
        )[0]

        hiddens.append(ht)
    
    # Stacking hidden states
    h = torch.stack(hiddens, dim=1)
    return h


class SLiGRU(Function):

    @staticmethod
    def forward(ctx, wx, ht, u, drop_mask):

        hiddens = []
        candidate_gate = []
        update_gate = []
        save_at = []
        save_mean = [] 
        save_rstd = []
        save_recurrent_gate = []
        ctx.h_init = ht
        eps = 1e-5
        normalized_shape = u.size(0)

        for t in range(wx.shape[1]):
            ht, hcand, zt_sig, at, recurrent_gate, mean,rstd= fast_sligru_cpp.forward(
                wx[:, t], 
                ht, 
                u, 
                drop_mask,
                normalized_shape,
                eps
            )

            hiddens.append(ht)
            candidate_gate.append(hcand)
            update_gate.append(zt_sig)
            save_at.append(at)
            save_mean.append(mean)
            save_rstd.append(rstd)
            save_recurrent_gate.append(recurrent_gate)

        ht = torch.stack(hiddens, dim=1)
        ctx.save_for_backward(wx, ht, u, drop_mask)

        ctx.candidate_gate = candidate_gate
        ctx.update_gate = update_gate
        ctx.save_at = save_at 
        ctx.save_mean = save_mean
        ctx.save_rstd = save_rstd
        ctx.save_recurrent_gate = save_recurrent_gate
        ctx.normalized_shape = normalized_shape
        return ht

    @staticmethod
    def backward(ctx, grad_out):
        wx, ht, u, drop_mask, = ctx.saved_tensors

        candidate_gate = ctx.candidate_gate
        update_gate = ctx.update_gate
        save_at = ctx.save_at 
        h_init = ctx.h_init 
        mean = ctx.save_mean
        rstd = ctx.save_rstd
        recurrent_gate = ctx.save_recurrent_gate
        normalized_shape = ctx.normalized_shape

        dh_prev = torch.zeros_like(ht[:, 0])
        du = torch.zeros_like(u)
        dwx = torch.zeros_like(wx)

        for t in reversed(range(wx.shape[1])):
            ht_ = h_init if t - 1 < 0 else ht[:, t - 1]

            dwx_, dh_prev, du = fast_sligru_cpp.backward(
                grad_out[:, t],
                dh_prev, 
                update_gate[t],
                save_at[t],
                drop_mask,
                ht_, 
                candidate_gate[t],
                u,
                du,
                recurrent_gate[t],
                mean[t],
                rstd[t],
                normalized_shape
            )

            dwx[:, t] = dwx_

        return dwx, None, du, None 


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
        torch.autograd.gradcheck(SLiGRU.apply, [wx, ht, u, drop_mask], atol=1e-5)
    

    out1 = ligru_cell(wx, u, ht, drop_mask)
    out2 = cuda_ligru_cell(normalized_shape, eps, drop_mask, wx, u, ht)
    assert torch.allclose(out1, out2, atol=10-5)
    
    warmup(ligru_cell, wx, u, ht, drop_mask, n_iters=10)
    print(benchmark(ligru_cell, wx, u, ht, drop_mask, n_iters=20))

        
    warmup(SLiGRU.apply, wx, ht, u, drop_mask, n_iters=10)
    print(benchmark(SLiGRU.apply, wx, ht, u, drop_mask, n_iters=20))