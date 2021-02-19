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

""" Code stolen from here: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""    
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
    
    
class CAModelConvLSTM(nn.Module):
    def __init__(self, num_states):
        super(CAModelConvLSTM, self).__init__()
        self.n_channels = num_states
        self.kernel_size = 3
        # First step rewrite transition using torch.nn.Unfold
        self.transition = nn.Sequential(
            nn.Linear(self.n_channels*self.kernel_size**2, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_channels) 
        )
        self.lstm = ConvLSTMCell(self.n_channels,self.n_channels, [3,3])
        self.output_lyr = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1)
        
            
    def forward_lstm(self, x, h, c):
        h, c = self.lstm(x,(h,c))
        return self.output_lyr(h), h, c
        
    def forward(self, x, steps=1):
        hx = torch.zeros(x.shape).to('cuda:0')
        cx = torch.zeros(x.shape).to('cuda:0')
                                 
        for _ in range(steps):
            x, hx, cx = self.forward_lstm(torch.softmax(x, dim=1), torch.softmax(hx, dim=1), torch.softmax(cx, dim=1))                   
        return x
    
class MetaConvLSTM_CA(nn.Module):
    def __init__(self, num_states):
        super(MetaConvLSTM_CA, self).__init__()
        self.n_channels = num_states
        self.kernel_size = 3
        # First step rewrite transition using torch.nn.Unfold
        self.transition = nn.Sequential(
            nn.Linear(self.n_channels*self.kernel_size**2, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_channels) 
        )
        
        self.conv_lstm_1 = ConvLSTMCell(self.n_channels,200, [3,3])
        #self.conv_lstm_2 = ConvLSTMCell(200,50, [5,5])
        self.lstm = nn.LSTMCell(200,200)
        self.output_lyr = nn.Conv2d(200, self.n_channels, kernel_size=1)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_states*2, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20*30*30, 200)
        )
        
            
    def forward_lstm(self, x, h1, c1):
        h1, c1 = self.conv_lstm_1(x,(h1,c1))
        return self.output_lyr(h1), h1, c1
        
    def forward(self, X, x, steps=1):
        h = torch.zeros([1,200]).to('cuda:0')
        c = torch.zeros([1,200]).to('cuda:0')
        X = self.cnn(X)
        for X_idx in range(X.shape[0]):
            input_X = X[X_idx].unsqueeze(0)
            
            h,c = self.lstm(input_X, (h,c))
        
        hx_1 = h.unsqueeze(-1).unsqueeze(-1).repeat([1,1,30,30])
        cx_1 = c.unsqueeze(-1).unsqueeze(-1).repeat([1,1,30,30])

        #hx.register_hook(lambda x: print(x.max(), x.min()))
        #cx.register_hook(lambda x: print(x.max(), x.min()))
        
        #hx_2 = torch.zeros([1,50,30,30]).to('cuda:0')
        #cx_2 = torch.zeros([1,50,30,30]).to('cuda:0')
                                
        for _ in range(steps):
            dx, hx_1, cx_1, = self.forward_lstm(torch.softmax(x, dim=1), hx_1, cx_1)
            x = x + dx 
        
        return x    
    
    def forward_imgs(self, X):
    
        """
        X: list[ Tensor[batch_size, n_channels,*], ..., ..]
            List with all the images in the examples of the task          
        """
        for x in X:
            print('asda')
            
            

        
    