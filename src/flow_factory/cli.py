# src/flow_factory/cli.py
"""
Command-line interface for Flow-Factory.
Acts as both the Supervisor (launcher) and the Worker (trainer).
"""
import sys
import os
import argparse
import subprocess
import logging
from typing import List

import torch
from .hparams.args import Arguments
from .trainers.loader import load_trainer
from .models.loader import load_model

# Configure basic logging for the CLI wrapper
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Flow-Factory Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (Auto-detects launcher from config)
    flow-factory-train config/flux_grpo.yaml
"""
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    # Allows parsing known args and ignoring the rest (handled by accelerate if needed)
    return parser.parse_known_args()

def run_training_logic(config):
    """The actual heavy lifting. Only called when ready to train."""
    from .trainers.loader import load_trainer
    trainer = load_trainer(config)
    trainer.run()

def train_cli():
    # 1. Load config
    args, unknown = parse_args() # Your existing parser
    config = Arguments.load_from_yaml(args.config)

    # 2. Determine if we need to launch the infrastructure
    # LLaMA-Factory logic: If GPUs > 1 and we aren't already a worker, launch torchrun/accelerate
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    is_distributed = os.environ.get("RANK") is not None
    
    if gpu_count > 1 and not is_distributed:
        logger.info("🚀 Launching distributed infrastructure...")
        subprocess.run([
            "accelerate", "launch",
            "--num_processes", str(gpu_count),
            "--main_process_port", str(config.main_process_port),
            sys.argv[0],  # Call this same script
            *sys.argv[1:] # Pass the same config and args
        ])
    else:
        # 3. Direct path: We are either on 1 GPU or already inside a worker
        logger.info("✅ Starting training execution...")
        run_training_logic(config)

if __name__ == "__main__":
    train_cli()