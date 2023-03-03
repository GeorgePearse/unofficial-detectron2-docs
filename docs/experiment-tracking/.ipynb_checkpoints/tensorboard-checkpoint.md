## Experiment Tracking with Tensorboard 

This comes out of the box when you train with Detectron2. I'd recommend setting the OUTPUT_DIR on the config, so that 
each run goes into its own directory, greatly simplifying run comparisons.

```python
clean_datetime = str(datetime.now()).replace(' ','_').replace(':','-')
cfg.OUTPUT_DIR = f'output/{clean_datetime}'
```

Limitations of Tensorboard: 
* Can't add notes to runs without further plugins.