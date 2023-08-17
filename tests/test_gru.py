import torch 
from fast_sligru import fast_gru, slow_gru


B, T, H, F = 4, 1, 10, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    inp_tensor = torch.rand([B, T, H], device=device)
    
    torch.manual_seed(42)
    net = slow_gru.GRU(hidden_size=H, input_shape=inp_tensor.shape).to(device)
    out_tensor_slow, _ = net(inp_tensor)
    
    torch.manual_seed(42)
    net = fast_gru.GRU(hidden_size=H, input_shape=inp_tensor.shape).to(device)
    out_tensor_fast, _ = net(inp_tensor)
    
    print("out_tensor_slow is = ", out_tensor_slow.sum())
    print("out_tensor_fast is = ", out_tensor_fast.sum())

    # cast to CPU 
    net.to("cpu")
    inp_tensor = inp_tensor.to("cpu")
    out_tensor_fast_cpu, _ = net(inp_tensor)
    print("out_tensor_fast_cpu is = ", out_tensor_fast_cpu.sum())
    assert torch.allclose(out_tensor_slow, out_tensor_fast, atol=1e-5), "Inference test failed!"
    print("Inference test passed!")
    


