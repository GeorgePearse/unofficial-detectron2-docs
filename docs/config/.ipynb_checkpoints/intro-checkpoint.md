## Configurable Parameters in Detectron2


Detectron2 is a config heavy API, really, this is one of its main strength. Train a diverse set of models without needing to edit any actual Python code.

That said there's an overwhelming number of options, it's hard to find out what many of them do, and it's often difficult to find what you're looking for.

## Model

```
cfg.MODEL.BACKBONE.FREEZE_AT = 2
```

How much of your model you want to be frozen.

```
cfg.MODEL.BACKBONE.FREEZE_AT = 0 
```

Would unfreeze your whole model. 

## Solver

### How to Drop the Learning Rate at set epochs