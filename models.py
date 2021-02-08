import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp



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

class LSTMCell:

    def __init__(self, input_size, hidden_size):
        self.input_size=input_size
        self.W_full = torch.FloatTensor(input_size + hidden_size, 
        hidden_size*4).type(dtype).to(device)
        init.normal(self.W_full, 0.0, 0.4)
        self.W_full = Variable(self.W_full, requires_grad = True)
        self.bias=1.0
        self.forget_bias=1.0

    def __call__(self, x, h, c):

        concat = torch.cat((x, h), dim=1)
        hidden = torch.matmul(concat, self.W_full)+self.bias

        i, g, f, o = torch.chunk(hidden, 4, dim=1)

        i = torch.sigmoid(i)
        g = torch.tanh(g)
        f = torch.sigmoid(f+self.forget_bias)
        o = torch.sigmoid(o)

        new_c = torch.mul(c, f) + torch.mul(g, i)
        new_h = torch.mul(torch.tanh(new_c), o)

        return new_h, new_c
    
    
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
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.convolve(torch.softmax(x, dim=1))
        return x
    
    def convolve(self, x, steps=1):

        windows = F.unfold(x, kernel_size=3, stride = 1, padding = 1)
        output = self.transition(windows.permute([2,0,1]))
        return output.permute(1,2,0).view(x.shape)
        
        
    