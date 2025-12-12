#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to run EvalAvalonWorkflow."""
import argparse
import sys
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentevolver.schema.task import Task
from games.games.avalon.workflows.eval_workflow import EvalAvalonWorkflow


def run_single_game(config_dict: Dict[str, Any], game_id: int, experiment_name: str = None, 
                    max_model_len: int = None, response_length: int = None) -> bool:
    """Run a single game workflow.
    
    Args:
        config_dict: Configuration dictionary for the game
        game_id: Unique identifier for this game instance
        experiment_name: Optional experiment name for organizing logs
        max_model_len: Maximum model length for formatter
        response_length: Response length for formatter
        
    Returns:
        bool: Result of the game (good_victory)
    """
    # Add experiment_name to config if provided
    if experiment_name:
        config_dict = config_dict.copy()
        config_dict['experiment_name'] = experiment_name
    
    # Add formatter config if provided
    if max_model_len is not None or response_length is not None:
        config_dict = config_dict.copy()
        if 'formatter' not in config_dict:
            config_dict['formatter'] = {}
        if max_model_len is not None:
            config_dict['formatter']['max_model_len'] = max_model_len
        if response_length is not None:
            config_dict['formatter']['response_length'] = response_length
    
    task = Task(
        task_id=f"eval_{game_id:03d}",
        env_type="",
        open_query=False,
        metadata={"avalon_config": config_dict},
    )
    
    workflow = EvalAvalonWorkflow(task=task)
    result = workflow.execute()
    return result


def main():
    parser = argparse.ArgumentParser(description="Run EvalAvalonWorkflow")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="task_config.yaml",
        help="Path to task config YAML file (default: task_config.yaml)",
    )
    parser.add_argument(
        "--num-games",
        "-n",
        type=int,
        default=1,
        help="Number of games to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: num_games)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing logs",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=25580,
        help="Maximum model length for formatter (default: None)",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=2048,
        help="Response length for formatter (default: None)",
    )
    args = parser.parse_args()
    
    # 解析配置文件路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config_path = config_path.resolve()
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # 读取 task_config.yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        print(f"Error: Config file is empty or invalid: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(config_dict, dict):
        print(f"Error: Config file must contain a dictionary, got {type(config_dict)}", file=sys.stderr)
        sys.exit(1)
    
    num_games = args.num_games
    max_workers = args.max_workers if args.max_workers is not None else num_games
    experiment_name = args.experiment_name
    max_model_len = args.max_model_len
    response_length = args.response_length
    
    if num_games == 1:
        result = run_single_game(config_dict, 0, experiment_name, max_model_len, response_length)
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_game, config_dict, game_id, experiment_name, max_model_len, response_length): game_id
                for game_id in range(num_games)
            }
            
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    result = future.result()
                    results.append((game_id, result))
                except Exception as e:
                    print(f"Game {game_id} failed: {e}", file=sys.stderr)
                    results.append((game_id, None))
        
        successful = sum(1 for _, r in results if r is not None)
        good_victories = sum(1 for _, r in results if r is True)
        if successful > 0:
            print(f"Summary: {good_victories}/{successful} good victories ({good_victories/successful*100:.1f}%)")
        else:
            print("Summary: All games failed", file=sys.stderr)


if __name__ == "__main__":
    main()

