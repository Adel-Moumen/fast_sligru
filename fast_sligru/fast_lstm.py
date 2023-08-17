""" This module implements a (SLOW) Gated Recurrent Units (GRU).

Author:
    * Adel Moumen 2023
"""
import torch.nn as nn 
import torch
from torch import Tensor
from typing import Optional

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
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
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
        """Initializes the layers of the liGRU."""
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = LSTM_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=0.0,
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
        """Returns the output of the liLSTM.
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

        # run liLSTM
        output, hh = self._forward_LSTM(x, hx=hx)

        return output, hh

    def _forward_LSTM(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla liLSTM.
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
        for i, liLSTM_lay in enumerate(self.rnn):
            if hx is not None:
                x = liLSTM_lay(x, hx=hx[i])
            else:
                x = liLSTM_lay(x, hx=None)
            if i != self.num_layers - 1:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h
    
    def compute_external_loss(self):
        total_loss = 0
        for layer in self.rnn:
            total_loss += layer.local_loss
        return total_loss / self.num_layers


class LSTM_Layer(torch.nn.Module):
    """ This function implements Light-Gated Recurrent Units (liLSTM) layer.
    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    batch_size : int
        Batch size of the input tensors.
    hidden_size : int
        Number of output neurons.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    normalization : str
        Type of normalization (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
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
        nonlinearity="relu",
        bidirectional=False,
    ):

        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.LSTM = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        if self.bidirectional:
            # Initial state
            self.register_buffer("h_init", torch.zeros(2, self.batch_size, self.hidden_size))
        else:
            self.register_buffer("h_init", torch.zeros(1, self.batch_size, self.hidden_size))

    def forward(self, x, hx: Optional[Tensor] = None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the LSTM layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """

        # Processing time steps
        if hx is not None:
            h = self._LSTM_cell(x, hx)
        else:
            h = self._LSTM_cell(x, self.h_init)

        return h

    def _LSTM_cell(self, x, ht):
        """Returns the hidden states for each time step.
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        # speechbrain case when we feed with a bs of 1 
        if x.shape[0] != ht.shape[1]:
            ht = torch.zeros(ht.shape[0], x.shape[0], ht.shape[2]).to(x.device)

        output, h = self.LSTM(x, ht)

        self.compute_local_loss(output.max())

        return output
    
    @torch.compile
    def _compute_lambda(self, norm_uz, norm_ur, norm_uh, max_value):

        # compute local loss
        lmbd = max_value / 4 * norm_uz + norm_ur + max_value / 4 * norm_uh

        return torch.abs(lmbd - 1)

    
    def compute_local_loss(self, max_value):
        # get recurrent weights
        ur, uz, uh = self.LSTM.weight_hh_l0.chunk(3, dim=0)

        # compute l2 norm
        norm_ur = torch.linalg.matrix_norm(ur, ord=2)
        norm_uz = torch.linalg.matrix_norm(uz, ord=2)
        norm_uh = torch.linalg.matrix_norm(uh, ord=2)


        self.local_loss = self._compute_lambda(
            norm_uz, norm_ur, norm_uh, max_value
        )
        


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