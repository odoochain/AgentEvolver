#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to run Avalon game evaluation with config-based workflow."""
import argparse
import sys
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from games.games.avalon.workflows.eval_workflow import EvalAvalonWorkflow
from games.utils import load_config


def build_task_configs(base_config: Dict[str, Any], num_games: int, 
                       experiment_name: str = None,
                       max_model_len: int = None, 
                       response_length: int = None) -> List[Dict[str, Any]]:
    """Build list of task configurations for parallel execution.
    
    Args:
        base_config: Base configuration dictionary
        num_games: Number of games to run
        experiment_name: Optional experiment name for organizing logs
        max_model_len: Maximum model length for formatter
        response_length: Response length for formatter
        
    Returns:
        List of configuration dictionaries, one per game
    """
    configs = []
    for game_id in range(num_games):
        config = copy.deepcopy(base_config)
        
        # Add experiment_name if provided
        if experiment_name:
            config['experiment_name'] = experiment_name
        
        # Add formatter config if provided
        if max_model_len is not None or response_length is not None:
            if 'formatter' not in config:
                config['formatter'] = {}
            if max_model_len is not None:
                config['formatter']['max_model_len'] = max_model_len
            if response_length is not None:
                config['formatter']['response_length'] = response_length
        
        configs.append(config)
    
    return configs


def run_single_game(config_dict: Dict[str, Any], game_id: int) -> Dict[str, Any]:
    """Run a single game workflow.
    
    Args:
        config_dict: Configuration dictionary for the game
        game_id: Unique identifier for this game instance (for logging)
        
    Returns:
        Dictionary containing game results
    """
    try:
        workflow = EvalAvalonWorkflow(config_dict=config_dict)
        result = workflow.execute()
        return result
    except Exception as e:
        print(f"Game {game_id} failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple games.
    
    Calculates average values for numeric keys in the result dictionaries.
    
    Args:
        results: List of result dictionaries from individual games
        
    Returns:
        Dictionary with aggregated statistics (averages, counts, etc.)
    """
    if not results:
        return {}
    
    # Filter out None results (failed games)
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return {"error": "All games failed"}
    
    # Collect all keys from all results
    all_keys = set()
    for result in valid_results:
        all_keys.update(result.keys())
    
    aggregated = {
        "total_games": len(results),
        "successful_games": len(valid_results),
        "failed_games": len(results) - len(valid_results),
    }
    
    # For each key, calculate average if values are numeric
    for key in all_keys:
        values = [r.get(key) for r in valid_results if key in r]
        
        # Skip None values
        numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
        
        if numeric_values:
            aggregated[f"{key}_mean"] = sum(numeric_values) / len(numeric_values)
            aggregated[f"{key}_sum"] = sum(numeric_values)
        
        # For list values like quest_results, handle specially
        if values and isinstance(values[0], list):
            # Average each position in the list
            max_len = max(len(v) for v in values if isinstance(v, list))
            if max_len > 0:
                position_sums = [0] * max_len
                position_counts = [0] * max_len
                for v in values:
                    if isinstance(v, list):
                        for i, val in enumerate(v):
                            if isinstance(val, (int, float)):
                                position_sums[i] += val
                                position_counts[i] += 1
                
                aggregated[f"{key}_position_avg"] = [
                    position_sums[i] / position_counts[i] if position_counts[i] > 0 else 0
                    for i in range(max_len)
                ]
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run EvalAvalonWorkflow")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/task_config.yaml",
        help="Path to task config YAML file (default: configs/task_config.yaml)",
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
        default=1,
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
    
    # 读取配置文件（支持继承）
    try:
        config_dict = load_config(config_path)
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(config_dict, dict):
        print(f"Error: Config file must contain a dictionary, got {type(config_dict)}", file=sys.stderr)
        sys.exit(1)
    
    num_games = args.num_games
    max_workers = args.max_workers if args.max_workers is not None else num_games
    experiment_name = args.experiment_name
    max_model_len = args.max_model_len
    response_length = args.response_length
    
    # Step 1: Build task list (configurations)
    print(f"Building {num_games} game configurations...")
    task_configs = build_task_configs(
        config_dict, 
        num_games, 
        experiment_name=experiment_name,
        max_model_len=max_model_len,
        response_length=response_length
    )
    
    # Step 2: Execute games in parallel
    print(f"Running {num_games} games (max_workers={max_workers})...")
    results = []
    
    if num_games == 1:
        # Single game execution
        result = run_single_game(task_configs[0], 0)
        results = [result]
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_game, task_configs[game_id], game_id): game_id
                for game_id in range(num_games)
            }
            
            for future in as_completed(futures):
                game_id = futures[future]
                result = future.result()
                results.append(result)
                if result is not None:
                    print(f"Game {game_id} completed: good_victory={result.get('good_victory')}")
    
    # Step 3: Aggregate and display results
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    
    aggregated = aggregate_results(results)
    
    # Display aggregated results
    for key, value in sorted(aggregated.items()):
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    print("="*60)
    
    # Display individual game results if only a few games
    if num_games <= 10:
        print("\nIndividual Game Results:")
        for i, result in enumerate(results):
            if result is not None:
                print(f"  Game {i}: good_victory={result.get('good_victory')}, "
                      f"quest_results={result.get('quest_results')}")
            else:
                print(f"  Game {i}: Failed")
    
    return aggregated


if __name__ == "__main__":
    main()

