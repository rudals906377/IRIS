"""StyleGAN2 fused_leaky_relu 의 순수 PyTorch 판(CUDA 확장 컴파일 없이 동작).

원본(rosinality/stylegan2-pytorch)은 import 때 CUDA 커널을 컴파일한다 → 윈도우에서는 Visual Studio C++ 와
CUDA 툴킷이 있어야 해서 설치가 자주 실패한다. 이 파일로 바꾸면 GPU·CPU 모두 컴파일 없이 돈다(조금 느림).
HairFastGAN 저장소의 models/stylegan2/op/fused_act.py 자리에 복사한다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale
    return F.leaky_relu(input, negative_slope=negative_slope) * scale
