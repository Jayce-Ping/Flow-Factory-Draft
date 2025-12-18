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
  flow-factory-train --config config/flux_grpo.yaml
  
  # Override launcher to run strictly locally
  # (Requires env config support or manual override logic)
  flow-factory-train --config config/flux_grpo.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    # Allows parsing known args and ignoring the rest (handled by accelerate if needed)
    return parser.parse_known_args()

def run_distributed_supervisor(config, args: List[str]):
    """
    The 'Supervisor': Constructs the infrastructure command and replaces the current process.
    """
    logger.info(f"🚀 [Flow-Factory Supervisor] Launching distributed training with mode: {config.launcher.upper()}")
    
    cmd = []

    # --- 1. Infrastructure Layer ---
    if config.launcher == "accelerate":
        cmd.extend(["accelerate", "launch"])
        
        # Distributed Config (e.g., deepspeed_stage2.yaml)
        if config.distributed_config_path:
            cmd.extend(["--config_file", config.distributed_config_path])
        
        # Process Count Override
        if config.num_processes > 1:
            cmd.extend(["--num_processes", str(config.num_processes)])
            
        # Add any other accelerate flags here if needed (e.g. main_process_port)
            
    elif config.launcher == "torchrun":
        cmd.extend(["torchrun"])
        cmd.extend(["--nproc_per_node", str(config.num_processes)])
        # Torchrun usually handles master_addr/port automatically in simple setups

    # --- 2. Application Layer ---
    # We call the exact same script entry point (e.g., 'flow-factory-train')
    # sys.argv[0] is the path to the script being executed
    cmd.append(sys.argv[0])
    
    # --- 3. Argument Layer ---
    # Pass through all original CLI arguments (e.g., --config x.yaml --debug)
    cmd.extend(args)
    
    logger.info(f"Executing Infrastructure Command: {' '.join(cmd)}")
    
    # Flush standard streams to prevent log interleaving
    sys.stdout.flush()
    sys.stderr.flush()
    
    # --- 4. Handoff ---
    # os.execvp replaces the current process image with the new command.
    # It does not return. Signals (Ctrl+C) are handled by the new process.
    try:
        os.execvp(cmd[0], cmd)
    except FileNotFoundError:
        logger.error(f"❌ Could not find executable: {cmd[0]}. Is 'accelerate' installed?")
        sys.exit(1)

def train_cli():
    """Train command entry point."""
    args, unknown = parse_args()
    
    # 1. Load Configuration (Lightweight)
    # We load this to check the 'env' settings
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = Arguments.load_from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

    # 2. Check Distributed Context
    # Accelerate/Torchrun set these variables. If they exist, we are a worker.
    # LOCAL_RANK is the most reliable indicator for Accelerate/Torchrun.
    is_worker = os.environ.get("LOCAL_RANK") is not None

    # Default to 'process' if env_args is missing or not set
    launcher_type = config.launcher

    # 3. Decision Logic
    if not is_worker and launcher_type != "process":
        # We are the User, and we want Distributed Training -> Start Supervisor
        # We pass sys.argv[1:] to forward --config, --resume, etc.
        run_distributed_supervisor(config, sys.argv[1:])

if __name__ == "__main__":
    train_cli()