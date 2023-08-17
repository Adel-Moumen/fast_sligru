
import torch
import torch.nn as nn 
import torch.autograd as autograd 

from typing import Optional, Tuple
from torch import Tensor


class LSTM(torch.nn.Module):
    """ 
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
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
        output, hh = self._forward_lstm(x, hx=hx)

        return output,  hh

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
                x = lstm_lay(x, hx=(hx[0][i], hx[1][i]))
            else:
                x = lstm_lay(x, hx=None)
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
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_size: int,
        dropout=0.0,
        bias=True,
        bidirectional=False,
    ):
        super(LSTM_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bias = bias
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(2, self.batch_size, self.hidden_size))
        self.register_buffer("c_init", torch.zeros(2, self.batch_size, self.hidden_size))


    def forward(self, x, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # type: (torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor] # noqa F821
        """Returns the output of the liGRU layer.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        
        # Processing time steps
        if hx is not None:
            h = self._lstm_cell(x, hx)
        else:
            h = self._lstm_cell(x, (self.h_init, self.c_init))

        return h

    def _lstm_cell(self, x, ht):
        """Returns the hidden states for each time step.
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        # speechbrain case when we feed with a bs of 1 
        if x.shape[0] != ht[0].shape[1]:
            h_n = torch.zeros(ht[0].shape[0], x.shape[0], ht[0].shape[2]).to(x.device)
            c_n = torch.zeros(ht[1].shape[0], x.shape[0], ht[1].shape[2]).to(x.device)

            output, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        else:
            output, (h_n, c_n) = self.lstm(x, ht)

        self.compute_local_loss(c_n.max())

        return output
    
    #@torch.compile
    def _compute_lambda(self, norm_ui, norm_uf, norm_ug, norm_uo, max_value):

        lmbd = 1/4 * norm_ui + max_value / 4 * norm_uf + norm_ug + 1/4 * norm_uo 

        return (lmbd - 1) ** 2

    
    def compute_local_loss(self, max_value):
        # get recurrent weights
        # (W_hi|W_hf|W_hg|W_ho)
        ui, uf, ug, uo = self.lstm.weight_hh_l0.chunk(4, dim=0)


        # compute l2 norm
        norm_ui = torch.linalg.matrix_norm(ui, ord=2)
        norm_uf = torch.linalg.matrix_norm(uf, ord=2)
        norm_ug = torch.linalg.matrix_norm(ug, ord=2)
        norm_uo = torch.linalg.matrix_norm(uo, ord=2)


        self.local_loss = self._compute_lambda(
            norm_ui, norm_uf, norm_ug, norm_uo, max_value
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

B, T, H, F = 4, 1, 10, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    inp_tensor = torch.rand([B, T, H], device=device)
    
    torch.manual_seed(42)
    net = LSTM(hidden_size=H, input_shape=inp_tensor.shape).to(device)
    out_tensor_slow, _ = net(inp_tensor)

    print(net.compute_external_loss())