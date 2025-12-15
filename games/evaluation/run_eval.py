#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified evaluation script for all games.

This script provides a unified entry point for evaluating different games.
It handles game selection, config loading, and result aggregation.

Basic Usage:
    python games/evaluation/run_eval.py \
        --game avalon \
        --config games/games/avalon/configs/task_config.yaml \
        --num-games 10 \
        --max-workers 5 \
        --experiment-name "my_experiment"

Using Local VLLM Models:
    To use local models with VLLM, you need to start the VLLM server separately first:
    
    Terminal 1 (start VLLM server):
        python games/evaluation/start_vllm.py --model-path /path/to/model --port 8000 --model-name local_model
    
    Terminal 2 (run evaluation):
        python games/evaluation/run_eval.py --game avalon --config games/games/avalon/configs/task_config.yaml --num-games 10
    
    Make sure your config file has the correct URL and model_name:
        default_model:
          url: http://localhost:8000/v1
          model_name: local_model
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from games.utils import load_config
from games.evaluation.eval_base import run_evaluation


# Game registry: maps game names to their evaluation functions
GAME_REGISTRY = {}


def register_game(name: str):
    """Decorator to register a game's evaluation function."""
    def decorator(func):
        GAME_REGISTRY[name] = func
        return func
    return decorator


def get_avalon_evaluator():
    """Get Avalon game evaluator function."""
    from games.games.avalon.workflows.eval_workflow import EvalAvalonWorkflow
    
    def run_single_game(config_dict: Dict[str, Any], game_id: int) -> Dict[str, Any]:
        """Run a single Avalon game."""
        try:
            workflow = EvalAvalonWorkflow(config_dict=config_dict)
            result = workflow.execute()
            return result
        except Exception as e:
            print(f"Game {game_id} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None
    
    return run_single_game


def get_diplomacy_evaluator():
    """Get Diplomacy game evaluator function."""
    # TODO: Update diplomacy workflow to return dict instead of using Task
    # For now, return a placeholder
    def run_single_game(config_dict: Dict[str, Any], game_id: int) -> Dict[str, Any]:
        """Run a single Diplomacy game."""
        # TODO: Implement when diplomacy workflow is updated
        raise NotImplementedError("Diplomacy evaluation needs to be updated to use config_dict")
    
    return run_single_game


# Register games
GAME_REGISTRY["avalon"] = get_avalon_evaluator()
GAME_REGISTRY["diplomacy"] = get_diplomacy_evaluator()


def display_results(aggregated: Dict[str, Any], game_name: str, num_games: int):
    """Display aggregated evaluation results.
    
    Args:
        aggregated: Aggregated results dictionary
        game_name: Name of the game
        num_games: Number of games run
    """
    print("\n" + "="*70)
    print(f"Evaluation Results Summary - {game_name.upper()}")
    print("="*70)
    
    # Display aggregated results
    for key, value in sorted(aggregated.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            if len(value) <= 10:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: [{value[0]}, ..., {value[-1]}] (length={len(value)})")
        else:
            print(f"  {key}: {value}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for all games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 Avalon games
  python games/evaluation/run_eval.py --game avalon --config games/games/avalon/configs/task_config.yaml --num-games 10
  
  # Run with custom experiment name
  python games/evaluation/run_eval.py --game avalon --config configs/task_config.yaml --num-games 5 --experiment-name "test_run"
        """
    )
    
    parser.add_argument(
        "--game",
        "-g",
        type=str,
        required=True,
        choices=list(GAME_REGISTRY.keys()),
        help=f"Game to evaluate. Choices: {', '.join(GAME_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to game config YAML file",
    )
    parser.add_argument(
        "--num-games",
        "-n",
        type=int,
        default=1,
        help="Number of games to run (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=1,
        help="Maximum number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing logs",
    )
    
    # Game-specific arguments (will be passed as kwargs)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length for formatter",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=None,
        help="Response length for formatter",
    )
    
    args = parser.parse_args()
    
    # Resolve config file path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to current directory first
        if not config_path.exists():
            # Try relative to game directory
            game_config_path = Path(__file__).parent.parent / "games" / args.game / args.config
            if game_config_path.exists():
                config_path = game_config_path
        config_path = config_path.resolve()
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load config file (supports Hydra inheritance)
    try:
        config_dict = load_config(config_path)
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(config_dict, dict):
        print(f"Error: Config file must contain a dictionary, got {type(config_dict)}", file=sys.stderr)
        sys.exit(1)
    
    # Get game evaluator
    run_single_game_fn = GAME_REGISTRY[args.game]
    
    # Prepare kwargs for additional config overrides
    kwargs = {}
    if args.max_model_len is not None:
        kwargs['formatter.max_model_len'] = args.max_model_len
    if args.response_length is not None:
        kwargs['formatter.response_length'] = args.response_length
    
    # Run evaluation
    try:
        aggregated = run_evaluation(
            game_name=args.game,
            config_dict=config_dict,
            num_games=args.num_games,
            max_workers=args.max_workers,
            experiment_name=args.experiment_name,
            run_single_game_fn=run_single_game_fn,
            **kwargs
        )
        
        # Display results
        display_results(aggregated, args.game, args.num_games)
        
        # Exit with error code if all games failed
        if aggregated.get("error") == "All games failed":
            sys.exit(1)
        
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

