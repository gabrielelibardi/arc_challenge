import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from torch.nn.parameter import Parameter



class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=5, padding=2),     
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(128, num_states, kernel_size=1)
        )
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x
    
    
    
class CAModelRecurrent(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=5, padding=2),     
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(128, num_states, kernel_size=1)
        )
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy
    
    
class CAModelMeta(nn.Module):
    def __init__(self, num_states, n_channels=11):
        super(CAModelMeta, self).__init__()
        self.n_channels = n_channels
        self.kernel_size = 3
        # First step rewrite transition using torch.nn.Unfold
        self.transition = nn.Sequential(
            nn.Linear(self.n_channels*self.kernel_size**2, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_channels) 
        )
        self.lstm = LSTMCell(self.n_channels, self.n_channels)
        self.output_lyr = nn.Sequential(nn.Linear(self.n_channels, self.n_channels))
            
        
    def forward_lstm(self, x, h,c):
        x = self.transition(x)
        h, c = self.lstm(x,(h,c))
        return self.output_lyr(h), h, c
        
    def forward(self, x, steps=1):
        hx = torch.zeros(x.shape)
        cx = torch.zeros(x.shape)
        for _ in range(steps):
            x, hx, cx = self.convolve(torch.softmax(x, dim=1), torch.softmax(hx, dim=1), torch.softmax(cx, dim=1))
        return x
    
    def convolve(self, x, h, c, steps=1):

        windows_x = F.unfold(x, kernel_size=3, stride = 1, padding = 1)
        windows_h = F.unfold(h, kernel_size=3, stride = 1, padding = 1)
        windows_c = F.unfold(c, kernel_size=3, stride = 1, padding = 1)
        y, h, c = self.forward_lstm(windows_x.permute([2,0,1]), windows_h.permute([2,0,1]), windows_c.permute([2,0,1]))
        return y.permute(1,2,0).view(x.shape), h.permute(1,2,0).view(x.shape), c.permute(1,2,0).view(x.shape)
        
        
    