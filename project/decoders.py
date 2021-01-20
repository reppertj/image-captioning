from warnings import warn
from torch import nn

BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'
PAD = '[PAD]'

class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_rnns,
        num_layers,
        nonlinearity,
        dropout,
        bidirectional,
        ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnns = num_rnns
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.bidirectional = bidirectional

        for i in range(num_rnns):
            setattr(
                self,
                "rnn%d" % i,
                nn.RNN(
                    self.input_size,
                    self.hidden_size,
                    num_layers=self.num_layers,
                    nonlinearity=nonlinearity,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

    def init_weights(self, method='kaiming'):
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            weight_fcn = lambda w: nn.init.kaiming_normal_(w)
        elif method == 'xavier':
            weight_fcn = lambda w: nn.init.xavier_normal_(w)
        
        bias_fcn = lambda b: nn.init.zeros_(b)

        for i in range(self.num_rnns):
            params = list(sum(zip(*getattr(self, "rnn%d" % i)._all_weights), ()))
            weights, biases = params[:(len(params) // 2)], params[(len(params) // 2):]
            list(map(weight_fcn,
                [getattr(self, "rnn%d" % i)._parameters[w] for w in weights]
            ))
            list(map(bias_fcn,
                [getattr(self, "rnn%d" % i)._parameters[b] for b in biases]
            ))

    def forward(self, wds, h0):
        rnn_outs = {}
        for i in range(self.num_rnns):
            rnn_outs["rnn%d" % i] = getattr(
                self,
                "rnn%d" % i
            )(wds[:, i, :, :], h0)
        return rnn_outs


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_rnns,
        num_layers,
        nonlinearity,
        dropout,
        bidirectional,
        ):
        super().__init__()

        if nonlinearity == 'relu':
            warn('GRU uses tanh and sigmoid internally; relu argument ignored')


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnns = num_rnns
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        for i in range(num_rnns):
            setattr(
                self,
                "rnn%d" % i,
                nn.GRU(
                    self.input_size,
                    self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

    def init_weights(self, method='kaiming'):
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            weight_fcn = lambda w: nn.init.kaiming_normal_(w)
        elif method == 'xavier':
            weight_fcn = lambda w: nn.init.xavier_normal_(w)
        
        bias_fcn = lambda b: nn.init.zeros_(b)

        for i in range(self.num_rnns):
            params = list(sum(zip(*getattr(self, "rnn%d" % i)._all_weights), ()))
            weights, biases = params[:(len(params) // 2)], params[(len(params) // 2):]
            list(map(weight_fcn,
                [getattr(self, "rnn%d" % i)._parameters[w] for w in weights]
            ))
            list(map(bias_fcn,
                [getattr(self, "rnn%d" % i)._parameters[b] for b in biases]
            ))

    def forward(self, wds, h0):
        rnn_outs = {}
        for i in range(self.num_rnns):
            rnn_outs["rnn%d" % i] = getattr(
                self,
                "rnn%d" % i
            )(wds[:, i, :, :], h0)
        return rnn_outs


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_rnns,
        num_layers,
        nonlinearity,
        dropout,
        bidirectional,
        ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnns = num_rnns
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if nonlinearity == 'relu':
            warn('LSTM uses tanh and sigmoid internally; relu argument ignored')

        for i in range(num_rnns):
            setattr(
                self,
                "rnn%d" % i,
                nn.LSTM(
                    self.input_size,
                    self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

    def init_weights(self, method='kaiming'):
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            weight_fcn = lambda w: nn.init.kaiming_normal_(w)
        elif method == 'xavier':
            weight_fcn = lambda w: nn.init.xavier_normal_(w)
        
        bias_fcn = lambda b: nn.init.zeros_(b)

        # This looks horrific but it's because of how standard pytorch nn
        # modules pack their weights and biases into a single, variable-sized
        # ordered dict, along with having a variable number of rnns in this
        # module
        for i in range(self.num_rnns):
            params = list(sum(zip(*getattr(self, "rnn%d" % i)._all_weights), ()))
            weights, biases = params[:(len(params) // 2)], params[(len(params) // 2):]
            list(map(weight_fcn,
                [getattr(self, "rnn%d" % i)._parameters[w] for w in weights]
            ))
            list(map(bias_fcn,
                [getattr(self, "rnn%d" % i)._parameters[b] for b in biases]
            ))

    def forward(self, wds, hn):
        rnn_outs = {}
        for i in range(self.num_rnns):
            rnn_outs["rnn%d" % i] = getattr(
                self,
                "rnn%d" % i
            )(wds[:, i, :, :], hn)
        return rnn_outs


class ParallelAttentionLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_features,
        num_heads,
        dropout,
        num_rnns,
        ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_rnns = num_rnns

        for i in range(num_rnns):
            setattr(
                self,
                "rnn%d" % i,
                AttentionLSTM(
                    input_size,  # wordvec_dim
                    hidden_size,
                    num_features,  # not including pixel dimensions, so just C from, e.g., (N, C, P, P)
                    num_heads,  # num // attention heads
                    dropout,
                )
            )        

    def init_weights(self, method='kaiming'):
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            weight_fcn = lambda w: nn.init.kaiming_normal_(w)
        elif method == 'xavier':
            weight_fcn = lambda w: nn.init.xavier_normal_(w)
        
        bias_fcn = lambda b: nn.init.zeros_(b)

        warn("Initialization not yet implemented for AttentionLSTM")

        # # This looks horrific but it's because of how standard pytorch nn
        # # modules pack their weights and biases into a single, variable-sized
        # # ordered dict, along with having a variable number of rnns in this
        # # module
        # for i in range(self.num_rnns):
        #     params = list(sum(zip(*getattr(self, "rnn%d" % i)._all_weights), ()))
        #     weights, biases = params[:(len(params) // 2)], params[(len(params) // 2):]
        #     list(map(weight_fcn,
        #         [getattr(self, "rnn%d" % i)._parameters[w] for w in weights]
        #     ))
        #     list(map(bias_fcn,
        #         [getattr(self, "rnn%d" % i)._parameters[b] for b in biases]
            # ))

    def forward(self, wds, feat, hn=None, cn=None):
        rnn_outs = {}
        if hn is None:
            for i in range(self.num_rnns):
                rnn_outs["rnn%d" % i] = getattr(
                    self,
                    "rnn%d" % i
                )(wds[:, i, :, :], feat)
        else:
            rnn_outs["rnn%d" % i] = getattr(
                self,
                "rnn%d" % i
            )(wds[:, i, :, :], feat, hn[:, i, :], cn[:, i, :])
        return rnn_outs

class AttentionLSTM(nn.Module):
    """ Context features via multihead attention before LSTM cell """
    def __init__(self,
        input_size,  # wordvec_dim
        hidden_size,
        num_features,  # not including pixel dimensions, so just C from, e.g., (N, C, P, P)
        num_heads,  # num // attention heads
        dropout,
        ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.dropout = dropout

        self.feature_pooling = nn.AdaptiveAvgPool2d(1)
        self.initial_hidden = nn.Linear(self.num_features, self.hidden_size)

        self.mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            kdim=self.num_features,
            vdim=self.num_features,
        )

        self.lstm = nn.LSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )

    def forward(self, wds, feat, hn=None, cn=None):
        """ wds: (batch_size, seq_len, input_size)
            feat: (batch_size, num_features, pixels, pixels)
            hn: (batch_size, hidden_dim)
            cn: (batch_size, hidden_dim)
        """
        if (hn is None and cn is not None) or (cn is not None and hn is None):
            raise ValueError('hidden and context matrices must be passed together')
        batch_size = wds.shape[0]
        out = wds.new(batch_size, wds.shape[1], self.hidden_size)
        key = feat.permute(2, 3, 0, 1).flatten(0, 1)  # (kdim, N, num_features)
        if hn is None:
            hn = self.initial_hidden(self.feature_pooling(feat).squeeze(3).squeeze(2)).unsqueeze(0)  # (1, N, hidden_size)
            cn = self.mha(key=key, value=key, query=hn, need_weights=False)[0].squeeze(0)  # (N, hidden_size)
            hn = hn.squeeze(0)  # (N, hidden_size)
        for i in range(wds.shape[1]):
            hn, cn = self.lstm(wds[:, i, :], (hn, cn))
            out[:, i, :] = hn
            hn = hn.unsqueeze(0)
            cn = self.mha(key=key, value=key, query=hn, need_weights=False)[0].squeeze(0)
            hn = hn.squeeze(0)
        return out, hn, cn


class ParallelFCScorer(nn.Module):
    def __init__(
        self,
        num_scorers,
        hidden_size,
        vocab_size,
        ):
        super().__init__()

        self.num_scorers = num_scorers

        for i in range(num_scorers):
            setattr(
                self,
                "fc%d" % i,
                nn.Linear(
                    hidden_size,
                    vocab_size,
                )
            )
        
    def init_weights(self, method='kaiming'):
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            weight_fcn = lambda fc: nn.init.kaiming_normal_(fc.weight)
        elif method == 'xavier':
            weight_fcn = lambda fc: nn.init.xavier_normal_(fc.weight)
        
        bias_fcn = lambda fc: nn.init.zeros_(fc.bias)

        for i in range(self.num_scorers):
            weight_fcn(getattr(self, "fc%d" % i))
            bias_fcn(getattr(self, "fc%d" % i))

    def forward(self, x):
        fc_outs = {}
        for i in range(self.num_scorers):
            fc_outs["fc%d" % i] = getattr(
                self,
                "fc%d" % i
            )(x["rnn%d" % i][0])
        return fc_outs

