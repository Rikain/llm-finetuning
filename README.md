For multi-gpu setup (if model training fits on one gpu and you want to speed it up running multiple gpus) run accelerate launch main.py.
Now using file named real-config.ini instead of config.ini that has the same contents.
T5 models have a known bug with fp16=True so they need fp16=False in config.
