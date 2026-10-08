from rl4co.tasks.train import train
import os
import time
import faulthandler

import torch
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["MPLBACKEND"] = "Agg"
if __name__ == "__main__":
    time_start = time.time()
    faulthandler.enable()
    train()
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
