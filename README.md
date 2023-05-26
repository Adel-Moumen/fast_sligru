`fast_sligru` is an open-source CUDA implementation that is the fastest public version of the [Stabilised Light Gated Recurrent Units](https://arxiv.org/abs/2302.10144). The implementation supports fp16, fp32, and fp64. It is based and compatible with PyTorch out of the box.

We benchmark the SLi-GRU on LibriSpeech and went from 7h19 to 2h33 of training time for one epoch on a single GPU (A100 80Gb).

For questions or feedback about `fast_sligru`, please open an issue on GitHub or send me an email at [adel.moumen@univ-avignon.fr](mailto:adel.moumen@univ-avignon.fr).

## GPU performance
Benchmarked on a single Tesla V100s (32Gb) (see: benchmark/benchmark.py). The improvements are more pronounced when 
running on longer sequences.

**Forward pass:**
| *Batch*=16, *Input size*=1024                 | fast SLi-GRU  | slow SLi-GRU (PyTorch) |
|-----------------------------------|-------|---------|
| *L*=100                           | 30.6 ms| 39.3 ms   |
| *L*=500                           | 118.0 ms| 145.5 ms   |
| *L*=1000                          | 185.3 ms| 270.0 ms   |
| *L*=2000                          | 339.9 ms| 526.3 ms   |
| *L*=3000                          | 486.1 ms| 800.5 ms   |


## How to use it 
```python
import torch
from fast_sligru import fast_sligru

batch, time, feats = 10, 10, 10
hidden_size, num_layer, dropout = 512, 4, 0.1

device = "cuda" # it works with CPU too

inp_tensor = torch.randn((batch, time, feats), requires_grad=False).to(device)
net = fast_sligru.SLiGRU( # or LiGRU
    input_shape=inp_tensor.shape,
    hidden_size=hidden_size,
    num_layers=num_layer,
    dropout=dropout,
).to(device)

# forward
out_tensor, _ = net(inp_tensor)

# backward
out_tensor.sum().backward()
```

### How to install 
To build the package:
```
pip install -r requirements.txt
pip install -e .
```

## Citing this work
To cite this work, please use the following BibTeX entry:
```
@inproceedings{moumen2023stabilising,
  title={Stabilising and accelerating light gated recurrent units for automatic speech recognition},
  author={Moumen, Adel and Parcollet, Titouan},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## License
[Apache 2.0](LICENSE)

