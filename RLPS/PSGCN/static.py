import torch 
from thop import clever_format, profile
from model import PSGCN

band = 176
classes = 13

model = PSGCN(band, classes)
input = torch.randn(1, 1024, band)
adj = torch.randn(classes, classes)
flops, params = profile(model, inputs=(input, adj))
flops, params = clever_format([flops, params], "%.2f")
print(flops)
print(params)