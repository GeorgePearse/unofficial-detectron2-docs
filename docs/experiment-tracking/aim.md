## Experiment Tracking with Aim

This was more or less a copy paste of the tracker written for MLFlow available here. 

https://philenius.github.io/machine%20learning/2022/01/09/how-to-log-artifacts-metrics-and-parameters-of-your-detectron2-model-training-to-mlflow.html

The one caveat is that their implementation did not support multi-gpu training.
That required a step to use 'comm' in order to check whether the process in which the code was being run was the 'main' process. Without this check an experiment was tracked 
for every GPU being used to run training e.g. you run python train.py and get an experiment for each GPU.

```
from detectron2.engine import HookBase
from aim import Run
import torch
import os
import detectron2.utils.comm as comm
from datetime import datetime

AIM_URL = os.environ["AIM_URL"]

class AimHook(HookBase):
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