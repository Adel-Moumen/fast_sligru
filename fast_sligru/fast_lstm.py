""" This module implements a CUDA Stabilised Light GRU (Li-GRU) cell.

Author:
    * Adel Moumen 2023
"""
from torch.autograd import Function
import torch.nn as nn 
import torch
import fast_sligru_cpp 
from torch import Tensor
from typing import Optional, Tuple

class LSTMCell(Function):
    """ This class implements a CUDA Stabilised Light GRU (Li-GRU) cell.

    SLi-GRU is single-gate GRU variant that uses a single gate to control the
    flow of information. 

    For more info see:
    "Moumen, A., & Parcollet, T. (2023, June). Stabilising and accelerating light gated recurrent units for automatic speech recognition.
    In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE."
    (https://arxiv.org/abs/2302.10144)
    """
    @staticmethod
    def forward(ctx, wx, ht, ct, u, drop_mask, training=False):
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
            
        if training:
            save_recurrent_gate = []
            save_input_gate = []
            save_forget_gate = []
            save_output_gate = []
            save_cell_gate = []


        ctx.h_init = ht
        ctx.training = training

        for t in range(wx.shape[1]):
            ht, ct, input_gate, cell_gate, forget_gate, output_gate, recurrent_gate, = fast_sligru_cpp.ligru_forward(
                wx[:, t], 
                ht, 
                ct,
                u, 
                drop_mask,
                training
            )

            hiddens.append(ht)
            # TODO: optimise this. If this is not training time then we can discard this and save memory
            if training:
                save_recurrent_gate.append(recurrent_gate)
                save_input_gate.append(input_gate)
                save_forget_gate.append(forget_gate)
                save_output_gate.append(output_gate)
                save_cell_gate.append(cell_gate)

        ht = torch.stack(hiddens, dim=1)

        if training:
            ctx.save_for_backward(wx, ht, u, drop_mask)
            ctx.save_input_gate = save_input_gate
            ctx.save_forget_gate = save_forget_gate
            ctx.save_output_gate = save_output_gate
            ctx.save_cell_gate = save_cell_gate
            ctx.save_recurrent_gate = save_recurrent_gate
            
        return ht, ct

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
                ctx.save_at[t],
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
class LSTM(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        input_shape,
        nonlinearity="relu",
        num_layers=1,
        bias=True,
        dropout=0.0,
        re_init=True,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.re_init = re_init
        self.bidirectional = bidirectional
        self.reshape = False
        self.nonlinearity = nonlinearity

        # Computing the feature dimensionality
        if len(input_shape) > 3:
            self.reshape = True
        self.fea_dim = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

        if self.re_init:
            rnn_init(self.rnn)

    def _init_layers(self):
        """Initializes the layers of the liGRU."""
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = LSTM_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                bidirectional=self.bidirectional,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the liGRU.
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

        # run lstm
        output, hh, ct = self._forward_lstm(x, hx=hx)

        return output, (hh, ct)

    def _forward_lstm(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla liGRU.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
        """
        h = []
        c = []
        if hx is not None:
            if self.bidirectional:
                hx[0] = hx[0].reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )

                hx[1] = hx[1].reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )
        # Processing the different layers
        for i, lstm_lay in enumerate(self.rnn):
            if hx is not None:
                x, ct = lstm_lay(x, hx=(hx[0][i], hx[1][i]))
            else:
                x, ct = lstm_lay(x, hx=None)
            h.append(x[:, -1, :])
            c.append(ct[:, -1, :])
        h = torch.stack(h, dim=1)
        c = torch.stack(c, dim=1)
        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
            c = c.reshape(c.shape[1] * 2, c.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)
            c = c.transpose(0, 1)

        return x, h, c



class LSTM_Layer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        bidirectional=False,
    ):
        super(LSTM_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Setting the activation function
        if nonlinearity == "tanh":
            self.act = torch.nn.Tanh()
        elif nonlinearity == "sin":
            self.act = torch.sin
        elif nonlinearity == "leaky_relu":
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.ReLU()

        self.w = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.u = nn.Linear(self.hidden_size, 4 * self.hidden_size)

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size))
        self.register_buffer("c_init", torch.zeros(1, self.hidden_size))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop()


    def forward(self, x, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # type: (torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor] # noqa F821
        """Returns the output of the liGRU layer.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)
        

        # Feed-forward affine transformations (all steps in parallel)
        wx = self.w(x)

        # Sampling dropout mask
        drop_mask = self._sample_drop_mask(wx)

        # Processing time steps
        if hx is not None:
            h, c, = hx 
            h, c = self._lstm_cell(wx, h, c, drop_mask)
        else:
            h, c = self._lstm_cell(wx, self.h_init, self.c_init, drop_mask)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

            c_f, c_b = c.chunk(2, dim=0)
            c_b = c_b.flip(1)
            c = torch.cat([c_f, c_b], dim=2)

        return h, c

    @torch.jit.ignore
    def _lstm_cell(self, w, ht, ct, drop_mask):
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
                # [H] -> [B, H] it makes the compiler happier
                drop_mask = drop_mask.repeat(w.shape[0], 1)
            h = LSTMCell.apply(w, ht, ct, self.u.weight, drop_mask, self.training)
        else:
            h = self._lstm_cell_cpu(w, ht, ct, drop_mask)
        return h
    

    def _lstm_cell_cpu(self, wx: torch.Tensor, ht: torch.Tensor, ct: torch.Tensor, drop_mask):
        """Returns the hidden states for each time step.
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        hiddens = []
        cell_state = []

        # Loop over time axis
        for k in range(wx.shape[1]):
            gates = wx[:, k] + self.u(ht)
            it, ft, gt, ot = gates.chunk(4, dim=-1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)

            ct = ft * ct + it * gt 
            ht = ot * self.act(ct) * drop_mask

            hiddens.append(ht)
            cell_state.append(ct)

        # Stacking states
        h = torch.stack(hiddens, dim=1)
        c = torch.stack(cell_state, dim=1)
        return h, c

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
            self.drop_mask_te = self.drop_mask_te.to(w.device)
            drop_mask = self.drop_mask_te

        return drop_mask

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
        )
        self.register_buffer("drop_mask_te", torch.tensor([1.0]).float())

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