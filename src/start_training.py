import logging
import time

import GPUtil

from src.full_model.train_full_model import main

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

gpus = GPUtil.getGPUs()
free_memory = gpus[0].memoryFree

while free_memory < 45000:
    time.sleep(10)
    log.info("Sleeping 10 seconds")

    gpus = GPUtil.getGPUs()
    free_memory = gpus[0].memoryFree

main()
