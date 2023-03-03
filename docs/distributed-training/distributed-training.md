## Distributed Training

Detectron2 only supports DDP (Distributed Data Parralel)

```
launch(
    run_training,
    num_gpus,
    num_machines=1,
    machine_rank=0,
    dist_url='auto',
    args=(cfg,),
)
```