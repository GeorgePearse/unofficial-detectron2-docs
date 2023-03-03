# Simplest Setup 


```python
register_coco_instances(
    "train", 
    {}, 
    cfg.training_annotations, 
    "<path-to-training-dataset>"
)

register_coco_instances(
    "val", 
    {}, 
    cfg.validation_annotations, 
    "<path-to-val-dataset>"
)
```


Register the datasets you'll use, and then specify them in the config.

```python
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
```