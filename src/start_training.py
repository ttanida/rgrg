import logging
import time

import GPUtil
import torch

# from src.full_model.train_full_model import main
from src.object_detector.training_script_object_detector import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

gpus = GPUtil.getGPUs()
free_memory = gpus[0].memoryFree

while free_memory < 26000:
    time.sleep(5)
    log.info("Sleeping 5 seconds")

    gpus = GPUtil.getGPUs()
    free_memory = gpus[0].memoryFree

# x = torch.rand(1024, 1024, 900 * 13, device=device)  # ^= 47651 MiB
# x = torch.rand(1024, 1024, 1024 * 11, device=device)  # ^= 45.9 GB
# x = torch.rand(1024, 1024, 1024 * 10, device=device)  # ^= 41.8 GB
# x = torch.rand(1024, 1024, 1024 * 9, device=device)  # ^= 37.7 GB
# x = torch.rand(1024, 1024, 1024 * 8, device=device)  # ^= 33.6 GB
# x = torch.rand(1024, 1024, 1024 * 7, device=device)  # ^= 29.5 GB
x = torch.rand(1024, 1024, 1024 * 6, device=device)  # ^= 25.4 GB
del x

main()
