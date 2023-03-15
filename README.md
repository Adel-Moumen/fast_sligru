`fast_ligru` is an open-source CUDA implementation that is the fastest public version of the [Light Gated Recurrent Units](https://arxiv.org/abs/1803.10225) and the [Stabilised Light Gated Recurrent Units](https://arxiv.org/abs/2302.10144) (5 times faster than a raw PyTorch implementation compiled with `torch.jit`). 

We provide two different implementations: `Li-GRU` and `SLi-GRU`. The difference lies in the recurrent connection: In `SLi-GRU` we apply layer normalisation on the recurrent weights to tackle down the gradient exploding problem. Indeed, `Li-GRU` is unstable and in practice cannot be trained on medium to large scale datasets (e.g, LibriSpeech 960h, CommonVoice) while `SLi-GRU` can and was designed for this purpose.

Both `Li-GRU` and `SLi-GRU` implementations support fp16, fp32 and fp64. Torch AMP also works.  Note that Layer Normalization, as used by `SLi-GRU`, disables fp16. This is intended.

These implementations were verified theoretically and empirically. We used `torch.autograd.gradcheck` and we scaled the `Li-GRUs` on real datasets including CommonVoice Italian, French, LibriSpeech 960h and TIMIT, where we got expected results. All the `Li-GRU`s on these datasets have been trained using [`SpeechBrain`, an all-in-one AI conversational toolkit](https://github.com/speechbrain/speechbrain/).

For questions or feedback about `fast_ligru`, please open an issue on GitHub or send me an email at [adel.moumen@univ-avignon.fr](mailto:adel.moumen@univ-avignon.fr).

## How to use it

```python
import torch
from fast_ligru_pytorch.ligru import LiGRU
from fast_ligru_pytorch.stabilised_ligru import SLiGRU

batch, time, feats = 10, 10, 10
hidden_size, num_layer, dropout = 512, 4, 0.1
nonlinearity = "relu" # works also with sine, leakyrelu and tanh

device = "cuda" # it works with CPU too

inp_tensor = torch.randn((batch, time, feats), requires_grad=False).to(device)
net = SLiGRU( # or LiGRU
    input_shape=inp_tensor.shape,
    hidden_size=hidden_size,
    num_layers=num_layer,
    dropout=dropout,
    nonlinearity=nonlinearity,
).to(device)

# forward
out_tensor, _ = net(inp_tensor)

# backward
out_tensor.sum().backward()
```

## Install

Here's what you'll need to get started:

- a [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) 3.7+ GPU (required)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0+ (required)
- [PyTorch](https://pytorch.org) 1.3+ for PyTorch integration

Once you have the prerequisites, you can install with pip or by building the source code.

### Building from source

```shell
make fast_ligru -j
```

install it with `pip`:

```shell
pip install fast_ligru-*.whl
pip install -e .
```

If the CUDA Toolkit that you're building against is not in `/usr/local/cuda`, you must specify the
`$CUDA_HOME` environment variable before running make, e.g.:

```shell
CUDA_HOME=/usr/local/cuda-10.2 make fast_ligru -j
```

## References

1. Ravanelli, M., Brakel, P., Omologo, M., & Bengio, Y. (2018). Light Gated Recurrent Units for Speech Recognition. arXiv. (<https://doi.org/10.1109/TETCI.2017.2762739>)

## Credits

The project is based on the [Haste](https://github.com/lmnt-com/haste) structure, but completely differs on the content implemented.

## Citing this work

To cite this work, please use the following BibTeX entry:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.10144,
  doi = {10.48550/ARXIV.2302.10144},
  
  url = {https://arxiv.org/abs/2302.10144},
  
  author = {Moumen, Adel and Parcollet, Titouan},
  
  keywords = {Audio and Speech Processing (eess.AS), Computation and Language (cs.CL), Machine Learning (cs.LG), Sound (cs.SD), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Stabilising and accelerating light gated recurrent units for automatic speech recognition},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

[Apache 2.0](LICENSE)
