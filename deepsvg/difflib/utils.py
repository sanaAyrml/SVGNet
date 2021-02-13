import torch
import matplotlib.pyplot as plt
import PIL.Image
import io









def is_clockwise(p):
    start, end = p[:-1], p[1:]
    return torch.stack([start, end], dim=-1).det().sum() > 0






