# src/utils/logger.py

import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Logger utility for console logs and TensorBoard.
    """
    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Args:
            log_dir (str): Base directory for logs/checkpoints.
            experiment_name (str): Optional subfolder name.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or timestamp
        self.log_path = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_path, exist_ok=True)

        # Set up console logger
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_path, "train.log"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Set up TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.log_path)

    def info(self, msg: str):
        """Log an info message to console and file."""
        self.logger.info(msg)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        self.tb_writer.add_scalar(tag, value, step)

    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()
