import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from torch.nn.parameter import Parameter


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        import ipdb; ipdb.set_trace()
        return x

class FlattenCustom(nn.Module):
    def __init__(self):
        super(FlattenCustom, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], 10, -1)


class NARAutoencoder(nn.Module):
    def __init__(self, num_states):
        super(NARAutoencoder, self).__init__()

        self.convolutions_stacks = nn.ModuleList([])
        self.n_channels = num_states
        self.width = 30 # input width
        self.height = 30
        kernels = [1,2,3,4,5,6,7,8,9,10]
        for kernel in kernels:
            # I don't really know the size of the out channels..
            # Think about the padding later
            # Not sure they are using Tanh intra-layers or only at the end
            transition = nn.Sequential(
                nn.Conv2d(num_states, 128, kernel_size=kernel, padding=0),     
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(128*(self.width -(kernel-1)) *(self.height-(kernel-1)), 256),
                nn.Tanh()
            )

            self.convolutions_stacks.append(transition)

        # So I guess the deciders output must be 10*256
        kernel_decider = 10
        self.decider = nn.Sequential(
                nn.Conv2d(num_states, 128, kernel_size=kernel_decider, padding=0),
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(128*(self.width -(kernel_decider-1)) *(self.height-(kernel_decider-1)), 2*10*256),
                nn.Tanh(),
                nn.Linear(2*10*256, 10*256),
                nn.Tanh(),
                FlattenCustom()
            )


    def forward(self, x, steps=1):

        # we don't know if the softmaz is really needed

        concat_conv_outputs = []
        for convolution_stack in self.convolutions_stacks:
            out_conv = convolution_stack(x)
            concat_conv_outputs.append(out_conv.unsqueeze(1))

        concat_conv_outputs_tensor = torch.cat(concat_conv_outputs, dim=1)
        masks = self.decider(x)
        return torch.mul(concat_conv_outputs_tensor, masks)
    
    
