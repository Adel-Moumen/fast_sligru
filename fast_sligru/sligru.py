from torch.autograd import Function
import torch
import fast_sligru_cpp 

class SLiGRUCell(Function):

    @staticmethod
    def forward(ctx, wx, ht, u, drop_mask, training=False):

        hiddens = []
        candidate_gate = []
        update_gate = []
        save_at = []
        save_mean = [] 
        save_rstd = []
        save_recurrent_gate = []
        ctx.h_init = ht
        ctx.training = training
        eps = 1e-5
        normalized_shape = u.size(0)

        for t in range(wx.shape[1]):
            ht, hcand, zt_sig, at, recurrent_gate, mean, rstd = fast_sligru_cpp.forward(
                wx[:, t], 
                ht, 
                u, 
                drop_mask,
                normalized_shape,
                eps,
                training
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
                normalized_shape,
                ctx.training
            )

            dwx[:, t] = dwx_

        return dwx, None, du, None 
