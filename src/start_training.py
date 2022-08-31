import logging
import time

import GPUtil
import torch

from src.full_model.train_full_model import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

gpus = GPUtil.getGPUs()
free_memory = gpus[0].memoryFree

while free_memory < 35000:
    time.sleep(5)
    log.info("Sleeping 5 seconds")

    gpus = GPUtil.getGPUs()
    free_memory = gpus[0].memoryFree

x = torch.rand(1024, 1024, 1024 * 11, device=device)  # ^= 45.9 GB
# x = torch.rand(1024, 1024, 1024 * 9, device=device)  # ^= 37.7 GB
del x

main()
