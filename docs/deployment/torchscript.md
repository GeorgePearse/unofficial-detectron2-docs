## Exporting a model to Torchscript

```
import torch

# Note, this is needed, despite not being explicitly called
import torchvision 

model = torch.jit.load('models/model.ts')
```