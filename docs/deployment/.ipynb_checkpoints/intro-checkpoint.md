## How to Deploy Models Trained with Detectron2

The repo comes with an unassuming script called export_model.py, it uses the rest of the package just as an API, and can be used as a standalone script or copied into your own repo (so that you don't have to clone detectron2).

It is overly verbose, so I've rewritten the core parts below with typer instead of python's default parser. 
It also just runs from a config.yaml (can obviously change the path to the weights here), but for my workflows that would normally point to the original weights that I started training from. Not my best checkpoint.
So I added an additional argument to point to those trained weights.

```
import typer 

def main(
        export_format: str = 'torchscript',
        architecture_name: str = 'R101',
        checkpoint_path: str = None,
    ): 
    DetectionCheckpointer()

if __name__ == '__main__':
    typer.run(main)
```


# Deployment

## Options 

* Torchscript 
- Gotchas 
- Make sure to import torchscript before reloading the saved model 
- Show what would be hit, show successful reload. 

* ONNX 
- Because it's a 'universal' framework, it offers thje most functionality wrt further optimizations (e.g. operator fusing or conversion to fp16 or int8) 

## Preprocessing 
Within detectron2 preprocessing is managed by a predictor object, but if you're using torchscript or ONNX, you're trying to remove your 
dependence on detectron2. In order to achieve this aim I simply extracted the preprocessing code from the predictor object.

