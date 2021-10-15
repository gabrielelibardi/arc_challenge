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
    def __init__(self, num_states, lamb, power_iterations):
        super(NARAutoencoder, self).__init__()

        self.n_channels = num_states
        self.width = 10 # input width
        self.height = 10
        kernels = [1,2,3,4,5,6,7,8,9,10]

        # ENCODER
        self.convolutions_stacks_enc = nn.ModuleList([])

        power_iterations = power_iterations
        lamb = lamb

        for kernel in kernels:
            # I don't really know the size of the out channels..
            # Think about the padding later
            # Not sure they are using Tanh intra-layers or only at the end
            transition = nn.Sequential(
                SpectralNorm(nn.Conv2d(num_states, 128, kernel_size=kernel, padding=0), power_iterations=power_iterations, lamb=lamb),     
                nn.Tanh(),
                nn.Flatten(),
                SpectralNorm(nn.Linear(128*(self.width -(kernel-1)) *(self.height-(kernel-1)), 256), power_iterations=power_iterations, lamb=lamb),
                nn.Tanh()
            )

            self.convolutions_stacks_enc.append(transition)

        # So I guess the deciders output must be 10*256
        kernel_decider = 10
        self.decider_enc = nn.Sequential(
                SpectralNorm(nn.Conv2d(num_states, 128, kernel_size=kernel_decider, padding=0), power_iterations=power_iterations, lamb=lamb),
                nn.Tanh(),
                nn.Flatten(),
                SpectralNorm(nn.Linear(128*(self.width -(kernel_decider-1)) *(self.height-(kernel_decider-1)), 128), power_iterations=power_iterations, lamb=lamb),
                nn.Tanh(),
                SpectralNorm(nn.Linear(128, 10*256)),
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
                SpectralNorm(nn.Linear(256, 128*(self.width -(kernel-1)) *(self.height-(kernel-1))), power_iterations=power_iterations, lamb=lamb),
                nn.Tanh(),
                nn.Unflatten(1, (128, (self.width -(kernel-1)), (self.height-(kernel-1)))),
                SpectralNorm(nn.ConvTranspose2d(128, num_states, kernel_size=kernel, padding=0), power_iterations=power_iterations, lamb=lamb),
                nn.Sigmoid(),
            )
            
            self.convolutions_stacks_dec.append(transition)

        # So I guess the deciders output must be 10*256
        kernel_decider = 10
        self.decider_dec = nn.Sequential(
                SpectralNorm(nn.Linear(256, 128)),
                nn.Tanh(),
                SpectralNorm(nn.Linear(128, 128*(self.width -(kernel_decider-1)) *(self.height-(kernel_decider-1))), power_iterations=power_iterations, lamb=lamb),
                nn.Tanh(),
                nn.Unflatten(1, (128, (self.width -(kernel-1)), (self.height-(kernel-1)))),
                SpectralNorm(nn.ConvTranspose2d(128, 10*num_states, kernel_size=kernel_decider, padding=0), power_iterations=power_iterations, lamb=lamb),
                nn.Tanh(),
                nn.Unflatten(1, (10, num_states))

            )


    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)

    def encode(self, x):
        concat_conv_outputs = []
        # Maybe this part could be done in parallel
        for convolution_stack in self.convolutions_stacks_enc:
            out_conv = convolution_stack(x)
            concat_conv_outputs.append(out_conv.unsqueeze(1))

        concat_conv_outputs_tensor = torch.cat(concat_conv_outputs, dim=1)
        masks = self.decider_enc(x)
        return torch.sum(torch.mul(concat_conv_outputs_tensor, masks), dim=1)

    def decode(self, x):
        concat_conv_outputs = []
        # Maybe this part could be done in parallel
        for convolution_stack in self.convolutions_stacks_dec:
            out_conv = convolution_stack(x)
            concat_conv_outputs.append(out_conv.unsqueeze(1))

        concat_conv_outputs_tensor = torch.cat(concat_conv_outputs, dim=1)
        masks = torch.softmax(self.decider_dec(x), dim=1)

        return torch.sum(torch.mul(concat_conv_outputs_tensor, masks), dim=1)




##### VQ-VAE ######


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings



class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, power_iterations, lamb):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            SpectralNorm(nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False), power_iterations=power_iterations, lamb = lamb),
            nn.ReLU(True),
            SpectralNorm(nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False), power_iterations=power_iterations, lamb = lamb)
            
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, power_iterations, lamb):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, power_iterations, lamb)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, power_iterations, lamb):
        super(Encoder, self).__init__()

        # changed padding from 1 to 2 
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=2)

        self._conv_1 =  SpectralNorm(self._conv_1, power_iterations=power_iterations, lamb = lamb)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 =  SpectralNorm(self._conv_2, power_iterations=power_iterations, lamb = lamb)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._conv_3 =  SpectralNorm(self._conv_3, power_iterations=power_iterations, lamb = lamb)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, 
                                             power_iterations=power_iterations, lamb = lamb)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)



class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, power_iterations, lamb):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)

        self._conv_1  = SpectralNorm(self._conv_1, power_iterations=power_iterations, lamb = lamb)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, 
                                             power_iterations=power_iterations, lamb = lamb)
        
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_trans_1 = SpectralNorm(self._conv_trans_1, power_iterations=power_iterations, lamb = lamb)
        
        # cahnged padding from 1 to 2 from original implementation, adn channel number changed from 3 sto 11
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=11,
                                                kernel_size=4, 
                                                stride=2, padding=2)

        self._conv_trans_2 = SpectralNorm(self._conv_trans_2, power_iterations=power_iterations, lamb = lamb)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0, power_iterations=1, lamb = 0.1):
        super(VQVAE, self).__init__()
        
        # channels number has been changed
        self._encoder = Encoder(11, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, 
                                power_iterations, lamb)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self._pre_vq_conv = SpectralNorm(self._pre_vq_conv, power_iterations=power_iterations, lamb = lamb)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                power_iterations, lamb)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity