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

while free_memory < 45000:
    time.sleep(10)
    log.info("Sleeping 10 seconds")

    gpus = GPUtil.getGPUs()
    free_memory = gpus[0].memoryFree

variable_size = 128
try:
    while free_memory < 45000:
        x = torch.rand(1024, 1024, variable_size, device=device)
        variable_size += 128

        gpus = GPUtil.getGPUs()
        free_memory = gpus[0].memoryFree
except Exception:
    variable_size -= 128
    x = torch.rand(1024, 1024, variable_size, device=device)

del x

main()
