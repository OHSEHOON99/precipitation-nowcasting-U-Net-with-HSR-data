import torch
import torch.nn as nn

class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, seq_len=6, device='cuda:0'):
        super(CGRU_cell, self).__init__()
        self.shape = shape  # H*W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.seq_len = seq_len
        self.padding = (filter_size - 1) // 2
        self.device = device
        # conv1: 
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None):
        # Move inputs and hidden state to the right device
        if inputs is not None:
            inputs = inputs.to(self.device)
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1], device=self.device)
        else:
            htprev = hidden_state.to(self.device)
            
        output_inner = []
        for index in range(self.seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1], device=self.device)
            else:
                x = inputs[index, ...]
                
            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev), 1)
            ht = torch.tanh(self.conv2(combined_2))
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, seq_len=6, device='cuda:0'):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.seq_len = seq_len
        self.padding = (filter_size - 1) // 2
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None):
        # Move inputs and hidden state to the right device
        if inputs is not None:
            inputs = inputs.to(self.device)
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1], device=self.device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1], device=self.device)
        else:
            hx, cx = hidden_state
            hx, cx = hx.to(self.device), cx.to(self.device)
            
        output_inner = []
        for index in range(self.seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1], device=self.device)
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)