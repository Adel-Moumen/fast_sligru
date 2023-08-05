import torch 
from fast_sligru import fast_ligru, fast_sligru



B, T, H, F = 4, 10, 10, 10


if __name__ == "__main__":
    
    torch.manual_seed(42)
    wx = torch.randn(B, T, H * 2, device="cuda", requires_grad=True, dtype=torch.double)
    ht = torch.randn(B, H, device="cuda", dtype=torch.double)
    u = torch.randn(H * 2, H, device="cuda", requires_grad=True, dtype=torch.double)
    drop_mask = torch.randn(B, H, device="cuda", dtype=torch.double)
    training=True


    assert torch.autograd.gradcheck(fast_ligru.LiGRUCell.apply, [wx, ht, u, drop_mask, training])

    print("gradcheck passed for fast_ligru.LiGRUCell.apply")

    assert torch.autograd.gradcheck(fast_sligru.SLiGRUCell.apply, [wx, ht, u, drop_mask, training])

    print("gradcheck passed for fast_sligru.SLiGRUCell.apply")
