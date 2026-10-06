
"""Model definitions and a single factory for all experiments."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)         
        self.activation = nn.ReLU()               
        self.fc2 = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        e = self.activation(self.fc1(x))

        return self.fc2(e)


class DNN1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)      
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()    
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.drop(self.activation(self.fc1(x)))

        return self.fc2(x)


class DNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1,  dropout_rate=0.25):
        super().__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.drop(self.activation(self.fc1(x)))
        x = self.drop(self.activation(self.fc2(x)))

        return self.fc3(x)


class DNN_Wide(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(self.activation(self.fc1(x)))
        return self.fc2(x)


class DNN_XtraWide(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, output_dim)

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(self.activation(self.fc1(x)))
        return self.fc2(x)


class DNN_Deep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(self.activation(self.fc1(x)))
        x = self.drop(self.activation(self.fc2(x)))
        return self.fc3(x)


class DNN_XtraDeep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, output_dim)

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(self.activation(self.fc1(x)))
        x = self.drop(self.activation(self.fc2(x)))
        x = self.drop(self.activation(self.fc3(x)))
        return self.fc4(x)


class FTTransformer(nn.Module):
    """
    FT-Transformer for tabular data.
    Designed to plug directly into your current training setup.
    Input:  (B, input_dim)
    Output: (B, 1)
    """

    def __init__(
        self,
        input_dim,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # project each feature -> token
        self.feature_proj = nn.Linear(1, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # column embeddings
        self.col_emb = nn.Embedding(input_dim + 1, d_model)

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        # prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):

        B = x.shape[0]

        # convert features -> tokens
        x = x.unsqueeze(-1)                     # (B, features, 1)
        tokens = self.feature_proj(x)           # (B, features, d_model)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # add column embeddings
        idx = torch.arange(tokens.size(1), device=x.device)
        tokens = tokens + self.col_emb(idx)

        # transformer
        z = self.encoder(tokens)

        # CLS readout
        rep = z[:, 0]

        rep = self.norm(rep)

        return self.head(rep)


class RNNClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNNCell(input_dim, hidden_dim)  # RNN Cell
        self.fc = nn.Linear(hidden_dim, output_dim)  # fully connected layer: maps last hidden vector to model prediction
        self.device = device


    def forward(self, x):
        x = x.to(self.fc.weight.dtype).to(self.device)
        hidden = self.init_hidden(x)
        
        time_steps = x.shape[1]                 # shape of x is (batches, time_steps, features)
        for i in range(0, time_steps):
            inputs = x[:,i]                     # shape of x is (batch, features) 
            hidden = self.rnn(inputs, hidden)
            
        out = self.fc(hidden)
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=self.device)
        return h0
    
    def forward_features(self, x):
        """
        Return penultimate features for RNNClassifier: final hidden state (B, hidden_dim)
        """
        x = x.to(self.fc.weight.dtype).to(self.device)
        hidden = self.init_hidden(x)
        time_steps = x.shape[1]
        for i in range(time_steps):
            inputs = x[:, i]
            hidden = self.rnn(inputs, hidden)
        # hidden is (B, hidden_dim)
        if hidden.dim() > 2:
            hidden = hidden.view(hidden.size(0), -1)
        return hidden


class LSTMClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)  # LSTM cell
        self.fc = nn.Linear(hidden_dim, output_dim)   # fully connected layer: maps last hidden vector to model prediction
        self.device = device


    def forward(self, x):
        x = x.to(self.fc.weight.dtype).to(self.device)
        hidden, cell = self.init_hidden(x)
        
        time_steps = x.shape[1]              # shape of x is (batches, time_steps, features)
        for i in range(0, time_steps):
            inputs = x[:,i]                  # shape of inputs is (batch, features)
            hidden, cell = self.rnn(inputs, (hidden,cell))
        
        out = self.fc(hidden)            
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=self.device)
        c0 = torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=self.device)
        return h0, c0
    
    def forward_features(self, x):
        """
        Penultimate features for LSTMClassifier: final hidden state (B, hidden_dim)
        """
        x = x.to(self.fc.weight.dtype).to(self.device)
        hidden, cell = self.init_hidden(x)
        time_steps = x.shape[1]
        for i in range(time_steps):
            inputs = x[:, i]
            hidden, cell = self.rnn(inputs, (hidden, cell))
        if hidden.dim() > 2:
            hidden = hidden.view(hidden.size(0), -1)
        return hidden


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()


class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding,True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv( x )
            out = self.batchnorm( out )
            out = self.chomp( out )
            return self.non_lin( out )


def _average_batch(x, lengths, B):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )


class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2, 
                 relu_type = 'relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len( kernel_sizes )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"



        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr( self,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)
        
        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr( self,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs//self.num_kernels) != n_outputs else None
        
        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr0_{}'.format(k_idx))
            outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )
                
        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]


            padding = [ (s-1)*dilation_size for s in self.ksizes]            
            layers.append( MultibranchTemporalBlock( in_channels, out_channels, self.ksizes, 
                stride=1, dilation=dilation_size, padding = padding, dropout=dropout, relu_type = relu_type,
                dwpw=dwpw) )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)        


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, 
                 symm_chomp = False, no_padding = False, relu_type = 'relu', dwpw=False):
        super(TemporalBlock, self).__init__()
        
        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2*padding-4
            padding = 1 # hack-ish thing so that we can use 3 layers

        if dwpw:
            self.net = nn.Sequential(
                # -- first conv set within block
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                # -- second conv set within block
                # -- dw
                nn.Conv1d( n_outputs, n_outputs, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=n_outputs, bias=False),
                nn.BatchNorm1d(n_outputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d( n_outputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm1 = nn.BatchNorm1d(n_outputs)
            self.chomp1 = Chomp1d(padding,symm_chomp)  if not self.no_padding else None
            if relu_type == 'relu':
                self.relu1 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu1 = nn.PReLU(num_parameters=n_outputs)
            self.dropout1 = nn.Dropout(dropout)
            
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm2 = nn.BatchNorm1d(n_outputs)
            self.chomp2 = Chomp1d(padding,symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu2 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu2 = nn.PReLU(num_parameters=n_outputs)
            self.dropout2 = nn.Dropout(dropout)
            
      
            if self.no_padding:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.relu2, self.dropout2)
            else:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size,True)
        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(TemporalConvNet, self).__init__()
        self.ksize = tcn_options['kernel_size'][0] if isinstance(tcn_options['kernel_size'], list) else tcn_options['kernel_size']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append( TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                     padding=(self.ksize-1) * dilation_size, dropout=dropout, symm_chomp = True,
                                     no_padding = False, relu_type=relu_type, dwpw=dwpw) )
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def forward_features(self, x):
        """
        Penultimate features for TemporalConvNet: returns trunk output pooled across time.
        Input expected (B, C_in, L) when used in TCN wrapper; if passed (B, L, C) caller should transpose.
        We mirror TCN.forward behaviour: trunk -> mean(dim=2)
        """
        # If input comes in (B, L, C) like other code, the caller should transpose before calling forward_features.
        out = self.network(x)
        # out shape (B, C_out, L)
        feats = out.mean(dim=2)  # (B, C_out)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        return feats


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, output_dim, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], output_dim)

        self.consensus_func = _average_batch

        self.has_aux_losses = False


    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        # x = self.consensus_func( x, lengths, B)
        x = x.mean(dim=2)
        x = self.tcn_output(x)
        return x
    
    def forward_features(self, x):
        """
        Return penultimate features for TCN: trunk output pooled over time (B, C_last).
        Matches TCN.forward before final linear.
        """
        x = self.tcn_trunk(x.transpose(1, 2))   # trunk output shape (B, C_last, L)
        feats = x.mean(dim=2)                   # (B, C_last)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        return feats


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, output_dim, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], output_dim)

        self.consensus_func = _average_batch


    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        # out = self.consensus_func( out, lengths, B )
        out = out.mean(dim=2)
        out = self.tcn_output(out)
        return out
    

    def forward_features(self, x):
        """
        Penultimate features for multiscale multibranch TCN:
        returns pooled trunk output (B, C_last).
        """
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)   # (B, C_last, L)
        feats = out.mean(dim=2)        # (B, C_last)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        return feats


class GaussianLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, funny = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        self.funny = funny
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # torch.nn.init.kaiming_normal_(self.weight, a=1 * np.sqrt(5))
        torch.nn.init.normal_(self.weight, 0, np.sqrt(2)/np.sqrt(self.in_features))
        # torch.nn.init.normal_(self.weight, 0, 3/np.sqrt(self.in_features))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)
            torch.nn.init.normal_(self.bias, 0, .1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class RNN_UntiedWeights(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, n_random_features, device='cuda'):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.act = nn.ReLU(inplace=True)
        self.device = device

        # Initialize RNN parameters without weight sharing
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(feature_dim, hidden_dim) for i in range(self.seq_len)
        ])

        # Initialise final linear layer
        self.linear = GaussianLinear(hidden_dim, n_random_features)

        # Initialize parameters with zero mean and unit variance
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.normal_(param.data, mean=0, std=1)
            else:
                nn.init.zeros_(param.data)

    def forward(self, input_seq):
        # returns the hidden state of the last layer and time step of the LSTM
        input_seq.to(self.device)
        batch_size, seq_len, _ = input_seq.size()
        h_t = torch.zeros(batch_size, self.hidden_dim).to(input_seq.device)

        for t in range(seq_len):
            x_t = input_seq[:, t, :]
            h_t = self.rnn_cells[t](x_t, h_t)
            h_t = self.act(h_t)  # Apply ReLU activation

        out = self.linear(h_t)
        return out


class LSTM_UntiedWeights(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, n_random_features, device='cuda'):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.act = nn.ReLU(inplace=True)
        self.device = device

        # Initialize LSTM parameters without weight sharing
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(feature_dim, hidden_dim) for i in range(self.seq_len)
        ])

        # Initialise final linear layer
        self.linear = GaussianLinear(hidden_dim, n_random_features)

        # Initialize parameters with zero mean and unit variance
        self.reset_parameters()
 
    def reset_parameters(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.normal_(param.data, mean=0, std=1)
            else:
                nn.init.zeros_(param.data)
    
    def forward(self, input_seq):
        # returns the hidden state of the last layer and time step of the LSTM
        input_seq = input_seq.to(self.device)
        batch_size, seq_len, _ = input_seq.size()
        h_t = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        for t in range(seq_len):
            x_t = input_seq[:, t, :]
            h_t, c_t = self.lstm_cells[t](x_t, (h_t, c_t))
            h_t = self.act(h_t)  # Apply ReLU activation

        out = self.linear(h_t)
        return out


_TCN_OPTIONS = {
    "tcn": {
        "relu_type": "prelu",
        "dropout": 0.5,
        "dwpw": False,
        "kernel_size": [3, 5, 7],
        "num_layers": 1,
        "width_mult": 1,
        "hidden_dim": 64,
    },
    "tcn2": {
        "relu_type": "prelu",
        "dropout": 0.75,
        "dwpw": False,
        "kernel_size": [9],
        "num_layers": 1,
        "width_mult": 1,
        "hidden_dim": 64,
    },
    "tcn3": {
        "relu_type": "prelu",
        "dropout": 0.5,
        "dwpw": False,
        "kernel_size": [3, 5],
        "num_layers": 2,
        "width_mult": 1,
        "hidden_dim": 128,
    },
}

DEFAULT_MODEL = {
    "eicu": "dnn",
    "mimic3_ihm": "tcn2",
    "mimic3_ph": "tcn2",
}


def default_net_type(dataset: str) -> str:
    try:
        return DEFAULT_MODEL[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset!r}") from exc


def get_model(
    dataset: str,
    net_type: str | None = None,
    device: str | torch.device = "cpu",
    *,
    output_dim: int | None = None,
) -> nn.Module:
    """Construct the architecture used by a dataset/task.

    ``dataset`` uses the unambiguous repo names ``eicu``, ``mimic3_ihm``
    and ``mimic3_ph``.  This prevents IHM and phenotype checkpoints from
    sharing the same ``mimic3`` directory/name.
    """
    dataset = dataset.lower()
    if dataset not in DEFAULT_MODEL:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    net_type = (net_type or DEFAULT_MODEL[dataset]).lower()

    if dataset == "eicu":
        out = output_dim or 1
        if net_type == "dnn":
            return DNN(402, 256, device, output_dim=out).to(device)
        if net_type == "dnn1":
            return DNN1(402, 256, output_dim=out).to(device)
        if net_type == "dnn2":
            return DNN2(402, 256, 128, output_dim=out).to(device)
        if net_type == "fttransformer":
            return FTTransformer(402, output_dim=out).to(device)
        raise ValueError(f"Unsupported eICU model: {net_type!r}")

    input_dim = 60
    default_output = 25 if dataset == "mimic3_ph" else 1
    out = output_dim or default_output

    if net_type in _TCN_OPTIONS:
        opts = deepcopy(_TCN_OPTIONS[net_type])
        num_channels = [
            opts["hidden_dim"] * len(opts["kernel_size"]) * opts["width_mult"]
        ] * opts["num_layers"]
        kwargs = {
            "input_size": input_dim,
            "output_dim": out,
            "num_channels": num_channels,
            "tcn_options": opts,
            "dropout": opts["dropout"],
            "relu_type": opts["relu_type"],
            "dwpw": opts["dwpw"],
        }
        cls = TCN if len(opts["kernel_size"]) == 1 else MultiscaleMultibranchTCN
        return cls(**kwargs).to(device)

    if net_type.startswith("lstm"):
        hidden = 128 if net_type == "lstm" else 256
        return LSTMClassifier(input_dim, hidden, device, output_dim=out).to(device)

    if net_type.startswith("rnn"):
        hidden = 128 if net_type == "rnn" else 256
        return RNNClassifier(input_dim, hidden, device, output_dim=out).to(device)

    raise ValueError(f"Unsupported model {net_type!r} for {dataset}.")
