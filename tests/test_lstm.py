import torch 
from fast_sligru import fast_lstm, slow_lstm


B, T, H, F = 4, 10, 10, 10


if __name__ == "__main__":
    
    inp_tensor = torch.rand([B, T, H])
    
    torch.manual_seed(42)
    net = slow_lstm.LSTM(hidden_size=H, input_shape=inp_tensor.shape)
    out_tensor_slow, _ = net(inp_tensor)
    
    torch.manual_seed(42)
    net = fast_lstm.LSTM(hidden_size=H, input_shape=inp_tensor.shape)
    out_tensor_fast, _ = net(inp_tensor)

    assert torch.allclose(out_tensor_slow, out_tensor_fast)
    print("Inference test passed!")
    


