""" This module implements a CUDA Stabilised Light GRU (Li-GRU) cell.

Author:
    * Adel Moumen 2023
"""
from torch.autograd import Function
import torch.nn as nn 
import torch
import fast_sligru_cpp 
from torch import Tensor
from typing import Optional

class LiGRUCell(Function):
    """ This class implements a CUDA Stabilised Light GRU (Li-GRU) cell.

    SLi-GRU is single-gate GRU variant that uses a single gate to control the
    flow of information. 

    For more info see:
    "Moumen, A., & Parcollet, T. (2023, June). Stabilising and accelerating light gated recurrent units for automatic speech recognition.
    In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE."
    (https://arxiv.org/abs/2302.10144)
    """
    @staticmethod
    def forward(ctx, wx, ht, u, drop_mask, training=False):
        """ This function implements the forward pass of the SLi-GRU cell.

        Arguments
        ---------
        wx : torch.Tensor
            The input tensor.
        ht : torch.Tensor
            The hidden state tensor.
        u : torch.Tensor
            The recurrent weight tensor.
        drop_mask : torch.Tensor
            The dropout mask tensor.
        training : bool
            Whether the model is in training mode or not.
        """

        hiddens = []
        candidate_gate = []
        update_gate = []
        save_at = []
        save_recurrent_gate = []
        
        ctx.h_init = ht
        ctx.training = training

        for t in range(wx.shape[1]):
            ht, hcand, zt_sig, at, recurrent_gate, = fast_sligru_cpp.ligru_forward(
                wx[:, t], 
                ht, 
                u, 
                drop_mask,
                training
            )

            hiddens.append(ht)
            candidate_gate.append(hcand)
            update_gate.append(zt_sig)
            save_at.append(at)
            save_recurrent_gate.append(recurrent_gate)

        ht = torch.stack(hiddens, dim=1)
        ctx.save_for_backward(wx, ht, u, drop_mask)

        ctx.candidate_gate = candidate_gate
        ctx.update_gate = update_gate
        ctx.save_at = save_at 
        ctx.save_recurrent_gate = save_recurrent_gate
        return ht

    @staticmethod
    def backward(ctx, grad_out):
        """ This function implements the backward pass of the SLi-GRU cell.

        Arguments
        ---------
        grad_out : torch.Tensor
            The gradient of the output.
        """
        wx, ht, u, drop_mask, = ctx.saved_tensors

        dh_prev = torch.zeros_like(ht[:, 0])
        du = torch.zeros_like(u)
        dwx = torch.zeros_like(wx)

        for t in reversed(range(wx.shape[1])):
            ht_ = ctx.h_init  if t - 1 < 0 else ht[:, t - 1]

            dwx_, dh_prev, du = fast_sligru_cpp.ligru_backward(
                grad_out[:, t],
                dh_prev, 
                ctx.update_gate[t],
                ctx.save_at [t],
                drop_mask,
                ht_, 
                ctx.candidate_gate[t],
                u,
                du,
                ctx.save_recurrent_gate[t],
                ctx.training
            )
            # CLIP if superior to 1
            # dh_prev = torch.clamp(dh_prev, max=1)
        
            dwx[:, t] = dwx_

        return dwx, None, du, None, None

class LiGRU(torch.nn.Module):
    """ This class implements a Stabilised Light GRU (Li-GRU).

    SLi-GRU is single-gate GRU model based on batch-norm + relu
    activations + layer-norm on the recurrent connections + recurrent dropout.

    The SLi-GRU differs from the vanilla Li-GRU on the recurrent weights. Indeed, the Li-GRU
    suffers from an exploding gradient problem on the recurrent weights, and cannot be trained on medium to large ASR dataset.
    To solve this problem, we use a layer-norm on the recurrent weights that stabilises the training of the model and allows one
    to train it on large ASR datasets without any problem.

    This model beat traditional LSTM/GRU models on the CommonVoice/LibriSpeech datasets (WER and efficiency).

    For more info see:
    "Moumen, A., & Parcollet, T. (2023, June). Stabilising and accelerating light gated recurrent units for automatic speech recognition.
    In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE."
    (https://arxiv.org/abs/2302.10144)

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).


    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    input_shape : tuple
        The shape of an example input.
    ff_normalization : str
        Type of feedforward normalization for the ligru model (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    recurrent_elementwise_affine : bool
        A boolean value that when set to True will enable the learnable affine parameters.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    bias : bool
        If True, the additive bias b is adopted.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    re_init : bool
        If True, orthogonal initialization is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = SLiGRU(input_shape=inp_tensor.shape, hidden_size=5)
    >>> out_tensor, _ = net(inp_tensor)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        ff_normalization="batchnorm",
        num_layers=1,
        bias=True,
        dropout=0.0,
        re_init=True,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ff_normalization = ff_normalization
        self.bias = bias
        self.dropout = dropout
        self.re_init = re_init
        self.bidirectional = bidirectional
        self.reshape = False

        # Computing the feature dimensionality
        if len(input_shape) > 3:
            self.reshape = True
        self.fea_dim = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

        if self.re_init:
            rnn_init(self.rnn)

    def _init_layers(self):
        """Initializes the layers of the SLi-GRU."""
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = SLiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                ff_normalization=self.ff_normalization,
                bias=self.bias,
                bidirectional=self.bidirectional,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the SLi-GRU.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor.
        hx : torch.Tensor
            Starting hidden state.
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run ligru
        output, hh = self._forward_sligru(x, hx=hx)

        return output, hh

    def _forward_sligru(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla SLi-GRU.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
        """
        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )
        # Processing the different layers
        for i, sligru_lay in enumerate(self.rnn):
            if hx is not None:
                x = sligru_lay(x, hx=hx[i])
            else:
                x = sligru_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h


class SLiGRU_Layer(torch.nn.Module):
    """ This class implements a Stabilised Light-Gated Recurrent Units (SLi-GRU) layer.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    num_layers : int
        The layer number.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    nonlinearity : str
        Type of nonlinearity (tanh, sin, leaky_relu, relu).
    ff_normalization : str
        Type of normalization (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in layer normalization.
        Note that this only applies to the feedforward affine transform.
        SLi-GRU (unlike Li-GRU) unconditionally applies layer normalization in
        the recurrent layers, which is unaffected by this parameter.
    bias: bool
        If True, the additive bias b is adopted.
    bidirectional : bool
        if True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        ff_normalization="batchnorm",
        bias=True,
        bidirectional=False,
    ):

        super(SLiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bias = bias

        self.w = nn.Linear(self.input_size, 2 * self.hidden_size, bias=False)

        self.u = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if ff_normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05)
            self.normalize = True

        elif ff_normalization == "layernorm":
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only  formally
            # initialized to avoid jit issues.
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size)
            self.normalize = True

        # we freeze the bias of the normalization layer
        if not self.bias:
            self.norm.bias.data.fill_(0)
            self.norm.bias.requires_grad = False

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop()

        self.act = torch.nn.ReLU()

    def forward(self, x, hx: Optional[Tensor] = None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the liGRU layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
            Hidden state.
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        if self.normalize:
            w_bn = self.norm(w.reshape(w.shape[0] * w.shape[1], w.shape[2]))
            w = w_bn.reshape(w.shape[0], w.shape[1], w.shape[2])


        # Sampling dropout mask
        drop_mask = self._sample_drop_mask(w)

        # Processing time steps
        if hx is not None:
            h = self._sligru_cell(w, hx, drop_mask)
        else:
            # broadcast to include batch size, this makes torch.compile happier
            h_init = self.h_init.broadcast_to(w.shape[0], self.h_init.shape[1])
            h = self._sligru_cell(w, h_init, drop_mask)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h
    
    def _sligru_cell_cpu(self, w, ht, drop_mask):
        hiddens = []

        for t in range(w.shape[1]):
            gates = w[:, t] + self.u(w)
            at, zt = gates.chunk(2, 1)
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand 
            hiddens.append(ht)
        
        # Stacking hidden states
        h = torch.stack(hiddens, dim=1)
        return h

    @torch.jit.ignore
    def _sligru_cell(self, w, ht, drop_mask):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        w : torch.Tensor
            Linearly transformed input.
        ht : torch.Tensor
            Hidden state.
        drop_mask : torch.Tensor
            Dropout mask.
        """
        if w.is_cuda:
            if not self.training:
                # [H] -> [B, H] it makes the compiler happy
                drop_mask = drop_mask.repeat(w.shape[0], 1)
            h = LiGRUCell.apply(w, ht, self.u.weight, drop_mask, self.training)
        else:
            h = self._sligru_cell_cpu(w, ht, drop_mask)
        return h

    def _init_drop(self):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size)).data,
            persistent=False,
        )
        self.register_buffer("drop_mask_te", torch.tensor([1.0]).float())

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks"""
        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=w.device
                    )
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            drop_mask = torch.ones(
                self.hidden_size, device=w.device
            )

        return drop_mask

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=x.device,
                    )
                ).data

def rnn_init(module):
    """This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.

    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)