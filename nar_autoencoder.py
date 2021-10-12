import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from torch.nn.parameter import Parameter

from spectral_normalization import SpectralNorm

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


class UnFlatten(nn.Module):
    def __init__(self):
        super(FlattenCustom, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], 10, -1)


class NARAutoencoder(nn.Module):
    def __init__(self, num_states):
        super(NARAutoencoder, self).__init__()

        self.n_channels = num_states
        self.width = 30 # input width
        self.height = 30
        kernels = [1,2,3,4,5,6,7,8,9,10]

        # ENCODER
        self.convolutions_stacks_enc = nn.ModuleList([])

        power_iterations = 5
        lamb = 0.1

        for kernel in kernels:
            # I don't really know the size of the out channels..
            # Think about the padding later
            # Not sure they are using Tanh intra-layers or only at the end
            transition = nn.Sequential(
                SpectralNorm(nn.Conv2d(num_states, 32, kernel_size=kernel, padding=0), power_iterations=power_iterations, lamb = lamb),     
                nn.Tanh(),
                nn.Flatten(),
                SpectralNorm(nn.Linear(32*(self.width -(kernel-1)) *(self.height-(kernel-1)), 256), power_iterations=power_iterations, lamb = lamb),
                nn.Tanh()
            )

            self.convolutions_stacks_enc.append(transition)

        # So I guess the deciders output must be 10*256
        kernel_decider = 10
        self.decider_enc = nn.Sequential(
                SpectralNorm(nn.Conv2d(num_states, 32, kernel_size=kernel_decider, padding=0), power_iterations=power_iterations, lamb = lamb),
                nn.Tanh(),
                nn.Flatten(),
                SpectralNorm(nn.Linear(32*(self.width -(kernel_decider-1)) *(self.height-(kernel_decider-1)), 2*10*32), power_iterations=power_iterations, lamb = lamb),
                nn.Tanh(),
                SpectralNorm(nn.Linear(2*10*32, 10*256)),
                nn.Tanh(),
                FlattenCustom()
            )

        # DECODER 
        self.convolutions_stacks_dec = nn.ModuleList([])

        for kernel in kernels:
            # I don't really know the size of the out channels..
            # Think about the padding later
            # Not sure they are using Tanh intra-layers or only at the end
            transition = nn.Sequential(
                SpectralNorm(nn.Linear(256, 32*(self.width -(kernel-1)) *(self.height-(kernel-1))), power_iterations=power_iterations, lamb = lamb),
                nn.Tanh(),
                nn.Unflatten(1, (32, (self.width -(kernel-1)), (self.height-(kernel-1)))),
                SpectralNorm(nn.ConvTranspose2d(32, num_states, kernel_size=kernel, padding=0), power_iterations=power_iterations, lamb = lamb),
                nn.Sigmoid(),
            )

            self.convolutions_stacks_dec.append(transition)

        # So I guess the deciders output must be 10*256
        kernel_decider = 10
        self.decider_dec = nn.Sequential(
                SpectralNorm(nn.Linear(256, 32)),
                nn.Tanh(),
                SpectralNorm(nn.Linear(32, 32*(self.width -(kernel_decider-1)) *(self.height-(kernel_decider-1))), power_iterations=power_iterations, lamb = lamb),
                nn.Tanh(),
                nn.Unflatten(1, (32, (self.width -(kernel-1)), (self.height-(kernel-1)))),
                SpectralNorm(nn.ConvTranspose2d(32, 10*num_states, kernel_size=kernel_decider, padding=0), power_iterations=power_iterations, lamb = lamb),
                nn.Tanh(),
                nn.Unflatten(1, (10, num_states))

            )



    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)

    def encode(self, x):
        concat_conv_outputs = []
        for convolution_stack in self.convolutions_stacks_enc:
            out_conv = convolution_stack(x)
            concat_conv_outputs.append(out_conv.unsqueeze(1))

        concat_conv_outputs_tensor = torch.cat(concat_conv_outputs, dim=1)
        masks = self.decider_enc(x)
        return torch.sum(torch.mul(concat_conv_outputs_tensor, masks), dim=1)

    def decode(self, x):
        concat_conv_outputs = []
        for convolution_stack in self.convolutions_stacks_dec:
            out_conv = convolution_stack(x)
            concat_conv_outputs.append(out_conv.unsqueeze(1))

        concat_conv_outputs_tensor = torch.cat(concat_conv_outputs, dim=1)
        masks = torch.softmax(self.decider_dec(x), dim=1)

        return torch.sum(torch.mul(concat_conv_outputs_tensor, masks), dim=1)
