import torchxrayvision as xrv
from torchinfo import summary
import torch.nn as nn

model = xrv.models.DenseNet(weights="densenet121-res224-all")
for name, child in model.named_children():
    print(name)
# summary(model, input_size=(64, 1, 224, 224))
