# -*- coding: utf-8 -*-

"""
@date: 2020/12/25
@file: test_model_complexity.py
@author: beiyan
@description: 
"""

import time
import torch
from thop import profile
from torchvision.models import resnet18


def compute_gflops_and_model_size(model):
    input = torch.randn(1, 3, 224, 224) # input size
    macs, params = profile(model, inputs=(input,), verbose=False)

    GFlops = macs * 2.0 / pow(10, 9)
    model_size = params * 4.0 / 1024 / 1024
    params_M = params/pow(10, 6)
    return params_M, model_size, GFlops

@torch.no_grad()
def compute_fps(model, shape, epoch=100, device=None):
    """
    frames per second
    :param shape: input size
    """
    total_time = 0.0

    if device:
        model = model.to(device)
    for i in range(epoch):
        data = torch.randn(shape)
        if device:
            data = data.to(device)

        start = time.time()
        outputs = model(data)
        end = time.time()

        total_time += (end - start)

    return total_time / epoch



def test_model_flops():
    model = resnet18()
    params_M, model_size, gflops = compute_gflops_and_model_size(model)

    print('Number of parameters: {:.2f} M '.format(params_M))
    print('Size of model: {:.2f} MB'.format(model_size))
    print('Computational complexity: {:.2f} GFlops'.format(gflops))

def test_fps():
    model = resnet18()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fps = compute_fps(model, (1, 3, 224, 224), device=device)
    print('device: {} - fps: {:.3f}s'.format(device.type, fps))


if __name__ == '__main__':
    test_model_flops()
    test_fps()
