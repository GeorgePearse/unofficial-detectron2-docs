## Experiment Tracking with MLFlow


```python
from detectron2.engine import HookBase
from aim import Run
import torch
import os
import detectron2.utils.comm as comm
from datetime import datetime

MLFLow_URL = os.environ["MLFlow_URL"]

class MLFlowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    
    All taken from https://philenius.github.io/machine%20learning/2022/01/09/how-to-log-artifacts-metrics-and-parameters-of-your-detectron2-model-training-to-mlflow.html
    And adapted for Aim
    
    Looking at write_metrics in this file can help with further development.
    https://github.com/facebookresearch/detectron2/blob/80e2673da161f57afe37ef769836a61976108ef1/detectron2/engine/train_loop.py#LL346
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        
        clean_datetime = str(datetime.now()).replace(' ','_').replace(':','-')
        
        # Have to check if it's the main process so that you dont 
        # get multiple tracked run for a single, multi-gpu process.s
        if comm.is_main_process():
            self.run = Run(
                repo=AIM_URL,
                experiment=clean_datetime,
            )
            self.run['hparams'] = self.cfg

    def after_step(self):
        # Only write metrics if it's the main process
        if comm.is_main_process():
            with torch.no_grad():
                latest_metrics = self.trainer.storage.latest()
                for k, v in latest_metrics.items():
                    self.run.track(name=k, value=v[0], step=self.trainer.storage.iter)

    def after_train(self):
        with torch.no_grad():
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
```

```
from detectron2.engine import HookBase
import detectron2.utils.comm as comm
import mlflow


class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        if comm.is_main_process():
            with torch.no_grad():
                mlflow.set_tracking_uri(self.cfg.MLFLOW.TRACKING_URI)
                mlflow.set_experiment(self.cfg.MLFLOW.EXPERIMENT_NAME)
                mlflow.start_run(run_name=self.cfg.MLFLOW.RUN_NAME)
                mlflow.set_tag("mlflow.note.content",
                               self.cfg.MLFLOW.RUN_DESCRIPTION)
                for k, v in self.cfg.items():
                    mlflow.log_param(k, v)

    def after_step(self):
        if comm.is_main_process():
            with torch.no_grad():
                latest_metrics = self.trainer.storage.latest()
                for k, v in latest_metrics.items():
                    mlflow.log_metric(key=k, value=v[0], step=v[1])

    def after_train(self):
        if comm.is_main_process():
            with torch.no_grad():
                with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                    f.write(self.cfg.dump())
                mlflow.log_artifacts(self.cfg.OUTPUT_DIR)
```

You then need to register the hook with your trainer

```
mlflow_hook = MLFlowHook(cfg)
trainer = DefaultTrainer()
trainer.register_hooks(hooks=[mlflow_hook])
```