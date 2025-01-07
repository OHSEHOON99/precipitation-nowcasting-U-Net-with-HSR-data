import logging
import torch
from collections import OrderedDict
from torch import nn
from .ConvRNN import CGRU_cell, CLSTM_cell


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4],
                                                 output_padding=1 if 'pad' in layer_name else 0)
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, *inputs.shape[1:]))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())  # Remove or conditionally log
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)
    
    
class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, *inputs.shape[1:]))
        return inputs

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in range(1, self.blocks)[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, seq_len=6, device='cuda:0'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len
        self.device = device
        self._initialize_rnn_cells()

    def _initialize_rnn_cells(self):
        """Initialize seq_len and device for all RNN cells in encoder and decoder."""
        for module in [self.encoder, self.decoder]:
            for name, layer in module.named_children():
                if isinstance(layer, (CGRU_cell, CLSTM_cell)):
                    layer.seq_len = self.seq_len
                    layer.device = self.device

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output
    
def find_rnn_type(module):
    for child in module.children():
        if isinstance(child, (CGRU_cell, CLSTM_cell)):
            return type(child).__name__
        rnn_type = find_rnn_type(child)
        if rnn_type:
            return rnn_type
    return None