#!/usr/bin/env python

import torch
from torch import nn
import numpy as np
import calnet.utils

def f_miller_troyer(mu,s,fudge=1e-4,use_s=True):
    # firing rate function, gaussian convolved with ReLU, derived in Miller and Troyer 2002
    if not use_s:
        s = s*0
    relu = torch.nn.ReLU()
    to_return = relu(mu)
    lkat = (s > fudge)
    u = mu[lkat]/torch.abs(s[lkat])/np.sqrt(2)
    A = 0.5*mu[lkat]*(1+torch.erf(u))
    B = torch.abs(s[lkat])/np.sqrt(2*np.pi)*torch.exp(-u**2)
    to_return[lkat] = A + B
    return to_return

def fprime_m_miller_troyer(mu,s,fudge=1e-4,use_s=True):
    # firing rate function, gaussian convolved with ReLU, derived in Miller and Troyer 2002
#     lkat = (s > fudge)
#     u = mu[lkat]/torch.abs(s[lkat])/np.sqrt(2)
#     A = 0.5*(1+torch.erf(u))
    if not use_s:
        s = s*0
    to_return = 0*mu
    to_return[mu > 0] = 1
    lkat = (s > fudge)
    u = mu[lkat]/torch.abs(s[lkat])/np.sqrt(2)
    A = 0.5*(1+torch.erf(u))
    to_return[lkat] = A
    return to_return

def f_miller_troyer_w0w1(mu_s):
    bdy = int(mu_s.shape[1]/2)
    Eta = mu_s[:,:bdy]
    Xi = mu_s[:,bdy:]
    return f_miller_troyer(Eta,Xi)

def f_miller_troyer_ws(mu_s,use_s=True):
    bdy = int(mu_s.shape[1]/4)
    Eta = mu_s[:,0:bdy] + mu_s[:,2*bdy:3*bdy]
    Xi = mu_s[:,bdy:2*bdy] + mu_s[:,3*bdy:4*bdy]
    return torch.cat((f_miller_troyer(Eta,Xi,use_s=use_s),fprime_m_miller_troyer(Eta,Xi,use_s=use_s)),axis=1)

class f_miller_troyer_ws_layer(nn.Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU.png
    """

    def __init__(self,use_s=True):#, output_channels: int):
        super().__init__()
        self.use_s = use_s
#         self.output_channels = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = f_miller_troyer_ws(input,use_s=self.use_s)
        return output

class drop_s_layer(nn.Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input),m(-input)))
    """

    def __init__(self):#, output_channels: int):
        super().__init__()
#         self.output_channels = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bdy = int(input.shape[1]/2)
        output = input[:,:bdy]
        return output
