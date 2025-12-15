# -*- coding: utf-8 -*-
"""Base evaluation framework for all games."""
import copy
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_task_configs(
    base_config: Dict[str, Any], 
    num_games: int,
    experiment_name: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Build list of task configurations for parallel execution.
    
    Args:
        base_config: Base configuration dictionary
        num_games: Number of games to run
        experiment_name: Optional experiment name for organizing logs
        **kwargs: Additional configuration overrides
        
    Returns:
        List of configuration dictionaries, one per game
    """
    configs = []
    for game_id in range(num_games):
        config = copy.deepcopy(base_config)
        
        # Add experiment_name if provided
        if experiment_name:
            config['experiment_name'] = experiment_name
        
        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if value is not None:
                if '.' in key:
                    # Handle nested keys like 'formatter.max_model_len'
                    keys = key.split('.')
                    current = config
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
                else:
                    # Handle flat keys
                    if key not in config:
                        config[key] = {}
                    if isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        
        configs.append(config)
    
    return configs


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
            aggregated[f"{key}_min"] = min(numeric_values)
            aggregated[f"{key}_max"] = max(numeric_values)
        
        # For list values, handle specially
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


def run_evaluation(
    game_name: str,
    config_dict: Dict[str, Any],
    num_games: int,
    max_workers: int = 1,
    experiment_name: Optional[str] = None,
    run_single_game_fn: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run evaluation for a game.
    
    Args:
        game_name: Name of the game (e.g., 'avalon', 'diplomacy')
        config_dict: Base configuration dictionary
        num_games: Number of games to run
        max_workers: Maximum number of parallel workers
        experiment_name: Optional experiment name for organizing logs
        run_single_game_fn: Function to run a single game. 
            Signature: (config_dict: Dict[str, Any], game_id: int) -> Dict[str, Any]
        **kwargs: Additional configuration overrides
        
    Returns:
        Aggregated results dictionary
    """
    if run_single_game_fn is None:
        raise ValueError(f"No run_single_game function provided for game: {game_name}")
    
    # Step 1: Build task list (configurations)
    print(f"[{game_name}] Building {num_games} game configurations...")
    task_configs = build_task_configs(
        config_dict,
        num_games,
        experiment_name=experiment_name,
        **kwargs
    )
    
    # Step 2: Execute games in parallel
    print(f"[{game_name}] Running {num_games} games (max_workers={max_workers})...")
    results = []
    
    if num_games == 1:
        # Single game execution
        result = run_single_game_fn(task_configs[0], 0)
        results = [result]
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_game_fn, task_configs[game_id], game_id): game_id
                for game_id in range(num_games)
            }
            
            for future in as_completed(futures):
                game_id = futures[future]
                result = future.result()
                results.append(result)
                if result is not None:
                    print(f"[{game_name}] Game {game_id} completed")
    
    # Step 3: Aggregate results
    aggregated = aggregate_results(results)
    
    return aggregated

