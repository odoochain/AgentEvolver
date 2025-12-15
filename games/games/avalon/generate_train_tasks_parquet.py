# -*- coding: utf-8 -*-
"""Generate training task Parquet file for Avalon game.

This script directly generates Parquet records without intermediate JSONL step.
"""
import argparse
import uuid
from pathlib import Path
import pandas as pd

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from games.utils import load_config


def generate_train_tasks_parquet(
    config_path: str,
    output_path: str,
    num_tasks: int = 10,
    env_type: str = "avalon",
    task_id_prefix: str = "avalon_train"
):
    """Generate training task Parquet file directly.
    
    Args:
        config_path: Path to task_config.yaml
        output_path: Path to output Parquet file
        num_tasks: Number of tasks to generate
        env_type: Environment type for tasks
        task_id_prefix: Prefix for task IDs
    """
    # Load config (supports inheritance)
    base_config = load_config(config_path)
    
    # Create avalon_config with assassin trainable
    avalon_config = {
        'default_model': base_config.get('default_model', {}),
        'game': base_config.get('game', {}),
        'roles': {'assassin': {'trainable': True}}
    }
    
    # Generate records directly
    records = []
    for i in range(num_tasks):
        task_id = f"{task_id_prefix}_{i:04d}"
        record = {
            "data_source": env_type,
            "prompt": [{"content": task_id, "role": "user"}],
            "reward_model": {"ground_truth": [], "style": "rule"},
            "uuid": str(uuid.uuid4()),
            "extras": {
                "task_id": task_id,
                "open_query": False,
                "new_query": None,
                "evaluator": "env",
                "ground_truth": None,
                "metadata": {"config": avalon_config}, # TODO: check if this is correct
            },
        }
        records.append(record)
    
    # Save to Parquet
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_file, index=False)
    
    print(f"Generated {num_tasks} tasks -> {output_file.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training task Parquet file")
    parser.add_argument("--config", type=str, default="games/avalon/configs/task_config.yaml", help="Path to task_config.yaml")
    parser.add_argument("--output", type=str, default="games/avalon/train_tasks.parquet", help="Path to output Parquet file")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to generate")
    parser.add_argument("--env_type", type=str, default="avalon", help="Environment type")
    parser.add_argument("--task_id_prefix", type=str, default="avalon_train", help="Prefix for task IDs")
    
    args = parser.parse_args()
    generate_train_tasks_parquet(
        config_path=args.config,
        output_path=args.output,
        num_tasks=args.num_tasks,
        env_type=args.env_type,
        task_id_prefix=args.task_id_prefix
    )


