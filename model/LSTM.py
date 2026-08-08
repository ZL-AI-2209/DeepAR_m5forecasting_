import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class VariationalDropout(nn.Module):
    def __init__(self, dropout: float, batch_first: bool = True):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)

        if is_packed:
            from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
            padded, lengths = pad_packed_sequence(x, batch_first=self.batch_first)
            padded = self.forward(padded)
            return pack_padded_sequence(padded, lengths, batch_first=self.batch_first, enforce_sorted=False)

        else:
            if self.batch_first:
                mask = x.new_empty(x.size(0), 1, x.size(-1), requires_grad=False).bernoulli_(1 - self.dropout)
            else:
                mask = x.new_empty(1, x.size(1), x.size(-1), requires_grad=False).bernoulli_(1 - self.dropout)

            return x.masked_fill(mask == 0, 0) / (1 - self.dropout)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropouti: float = 0., dropouto: float = 0., dropoutw: float = 0.,
                 batch_first: bool = True, unit_forget_bias: bool = True, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropoutw = dropoutw

        self.dropout_input = VariationalDropout(dropout=dropouti, batch_first=batch_first)
        self.dropout_output = VariationalDropout(dropout=dropouto, batch_first=batch_first)

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropoutw if num_layers > 1 else 0.0,
            batch_first=batch_first,
            **kwargs
        )

        self.unit_forget_bias = unit_forget_bias
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

                if self.unit_forget_bias and "bias_ih" in name:
                    with torch.no_grad():
                        param[self.hidden_size: 2 * self.hidden_size].fill_(1.0)

    def forward(self, input: torch.Tensor, hx=None):
        input_dropped = self.dropout_input(input)
        seq, state = self.rnn(input_dropped, hx)
        seq_dropped = self.dropout_output(seq)

        return seq_dropped, state