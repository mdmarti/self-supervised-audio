# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#### From torch's documentation:

### size of 1d conv output will be:
# L_out = floor([L_in + 2 x pad - dilation x (kernel - 1) - 1]/[stride] + 1)

def conv1d(in_channels,out_channels,kernel_size,stride=1,dilation=1):

    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=False,
        groups=1,
        padding=dilation
    )


class Encoder(nn.Module):

    def __init__(
            self,
            in_channels,
            dilations=[2,4,8,8,4],
            strides=[5,4,2,2,2],
            kernels=[10,8,4,4,4],
            encode_dim=32,
            norm_layer=None,
    ):
        super(Encoder,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.encode_dim = encode_dim

        encode_dim //= 2
        

        self.strided_encoder = nn.Sequential(
            conv1d(in_channels, # downsample to len 87
                   out_channels=encode_dim,
                   kernel_size=kernels[0],
                   stride=strides[0],
                   dilation=1),
                   norm_layer(encode_dim),
            conv1d(encode_dim, # downsample to len 21
                   out_channels=encode_dim,
                   kernel_size=kernels[1],
                   stride=strides[1],
                   dilation=1),
                   norm_layer(encode_dim),
            conv1d(encode_dim, # downsample to len 10
                   out_channels=encode_dim,
                   kernel_size=kernels[2],
                   stride=strides[2],
                   dilation=1),
                   norm_layer(encode_dim),
            conv1d(encode_dim, # downsample to len 4
                   out_channels=encode_dim,
                   kernel_size=kernels[3],
                   stride=strides[3],
                   dilation=1),
                   norm_layer(encode_dim),
            conv1d(encode_dim, #downsample to len 1
                   out_channels=encode_dim,
                   kernel_size=kernels[4],
                   stride=strides[4],
                   dilation=1),
                   norm_layer(encode_dim)
            )

        self.dilated_encoder=nn.Sequential(
             conv1d(in_channels, # downsample to len 87
                   out_channels=encode_dim,
                   kernel_size=kernels[0],
                   stride=3,
                   dilation=dilations[0]),
                   norm_layer(encode_dim),
            conv1d(encode_dim, # downsample to len 21
                   out_channels=encode_dim,
                   kernel_size=kernels[1],
                   stride=3,
                   dilation=dilations[1]),
                   norm_layer(encode_dim),
            conv1d(encode_dim, # downsample to len 10
                   out_channels=encode_dim,
                   kernel_size=kernels[2],
                   stride=2,
                   dilation=dilations[2]),
                   norm_layer(encode_dim),
            conv1d(encode_dim, # downsample to len 4
                   out_channels=encode_dim,
                   kernel_size=kernels[3],
                   stride=2,
                   dilation=dilations[3]),
                   norm_layer(encode_dim),
            conv1d(encode_dim, #downsample to len 1
                   out_channels=encode_dim,
                   kernel_size=kernels[4],
                   stride=3,
                   dilation=dilations[4]),
                   norm_layer(encode_dim)
            )
        
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self,x):

        dilatedRep,stridedRep = self.dilated_encoder(x),self.strided_encoder(x)
        rep = torch.cat([dilatedRep.squeeze(-1),stridedRep.squeeze(-1)],dim=-1)
        return rep
        
def Projector(sizes):

    layers = []
    for ii in range(len(sizes) - 2):

        layers.append(nn.Linear(sizes[ii],sizes[ii+1]))
        layers.append(nn.BatchNorm1d(sizes[ii+1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(sizes[-2],sizes[-1],bias=False))
    return nn.Sequential(*layers)

class VicRegWaves(nn.Module):

    def __init__(self,args,dilations,strides,kernels,encode_dim,sizes,sim_coeff=1,std_coeff=1,cov_coeff=1):
        super().__init__()
        self.args = args 
        self.encoder = Encoder(in_channels=1,
                               dilations=dilations,
                               strides=strides,
                               kernels=kernels,
                               encode_dim=encode_dim,
                               norm_layer=nn.BatchNorm1d)
        self.projector = Projector(sizes)
        self.latent_dim=encode_dim
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self,x1,x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z2Hat = self.projector(z1)

        repr_loss = F.mse_loss(z2Hat,z2)
        
        centeredZ2 = z2 - z2.mean(dim=0)
        centeredZ2hat = z1 - z1.mean(dim=0)

        std_z2 = torch.sqrt(centeredZ2.var(dim=0) + 1e-4)
        std_z2hat = torch.sqrt(centeredZ2hat.var(dim=0) + 1e-4)

        std_loss = torch.mean(F.relu(1 - std_z2))/2 + torch.mean(F.relu(1-std_z2hat))/2
        
        cov_z2 = centeredZ2.T @ centeredZ2 / centeredZ2.shape[0]
        cov_z2hat = centeredZ2hat.T @ centeredZ2hat / centeredZ2hat.shape[0]
        
        cov_loss = off_diagonal(cov_z2).pow_(2).sum().div(self.latent_dim) + \
            off_diagonal(cov_z2hat).pow_(2).sum().div(self.latent_dim)

        loss = (self.sim_coeff*repr_loss 
                + self.std_coeff * std_loss 
                + self.cov_coeff * cov_loss)
        
        return loss 
    
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
