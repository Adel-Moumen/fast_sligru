`fast_sligru` is an open-source CUDA implementation that is the fastest public version of the [Stabilised Light Gated Recurrent Units](https://arxiv.org/abs/2302.10144). The implementation supports fp16, fp32, and fp64. It is based and compatible with PyTorch out of the box.

We benchmark the SLi-GRU on LibriSpeech and went from 7h19 to 2h33 of training time for one epoch on a single GPU (A100 80Gb).

For questions or feedback about `fast_sligru`, please open an issue on GitHub or send me an email at [adel.moumen@univ-avignon.fr](mailto:adel.moumen@univ-avignon.fr).

## GPU performance
Benchmarked on a single Tesla V100s (32Gb) (see: benchmark/benchmark.py). The improvements are more pronounced when 
running on longer sequences. Each SLi-GRU is composed of 4 layers of 512 units. The input size is 1024. The batch size is 16.

**Forward pass:**
| *Batch*=16                 | fast SLi-GRU (CUDA+PyTorch) | slow SLi-GRU (PyTorch) |
|-----------------------------------|-------|---------|
| *L*=100                           | 0.05 s| 0.11 s   |
| *L*=500                           | 0.25 s| 0.55 s   |
| *L*=1000                          | 0.50 s| 1.11 s   |
| *L*=2000                          | 1.02 s| 2.26 s   |
| *L*=3000                          | 1.55 s| 3.39 s   |


**Backward pass:**
| *Batch*=16                 | fast SLi-GRU (CUDA+PyTorch) | slow SLi-GRU (PyTorch) |
|-----------------------------------|-------|---------|
| *L*=100                           | 0.15 s| 0.25 s   |
| *L*=500                           | 0.63 s| 1.29 s   |
| *L*=1000                          | 1.27 s| 3.68 s   |
| *L*=2000                          | 2.65 s| 11.87 s   |
| *L*=3000                          | 3.84 s| 24.39 s   |

## How to use it with SpeechBrain
You just have to change the name of the RNN class in your yaml file.
See ASR seq2seq LibriSpeech: [SpeechBrain](
  https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/seq2seq/hparams/train_BPE_1000.yaml#L89
)
```yaml
# in your yaml file
rnn_class: !name:fast_sligru.fast_ligru.LiGRU
# or if you want to use the SLi-GRU
rnn_class: !name:fast_sligru.fast_sligru.SLiGRU
```

## How to use it with PyTorch
```python
import torch
from fast_sligru import fast_sligru

batch, time, feats = 10, 10, 10
hidden_size, num_layer, dropout = 512, 4, 0.1

device = "cuda" # it works with CPU too

inp_tensor = torch.randn((batch, time, feats), requires_grad=False).to(device)
net = fast_sligru.SLiGRU( 
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
```bash
git clone https://github.com/Adel-Moumen/fast_sligru.git
cd fast_sligru/
pip install -r requirements.txt
pip install -e .
```

## Citing this work
To cite this work, please use the following BibTeX entry:
```
@INPROCEEDINGS{10095763,
  author={Moumen, Adel and Parcollet, Titouan},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Stabilising and Accelerating Light Gated Recurrent Units for Automatic Speech Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095763}
}
```

## License
[Apache 2.0](LICENSE)

