## Exporting a model to Torchscript

```
import torch

# Note, this is needed, despite not being explicitly called
import torchvision 

model = torch.jit.load('models/model.ts')
```

torchscript model sizes

* R50 -> 158 MB
* R101 -> 230 MB 
* X101 -> 400 MB
