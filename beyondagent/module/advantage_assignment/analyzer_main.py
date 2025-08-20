#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime
import argparse

# 项目根目录进 sys.path（按你原来的方式）
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from beyondagent.module.advantage_assignment.advantage_analyzer import analyze_single_checkpoint

def main():
    parser = argparse.ArgumentParser(
        description="Advantage Analyzer (no hardcoded params; forward all Hydra overrides)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory that contains the 'actor' subfolder (e.g., .../global_step_50)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Hydra YAML config file (e.g., config/beyond_agent_dataflow.yaml)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Where to save results; default: <cwd>/analysis_results/<timestamp>",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--enable-semantic", action="store_true", default=True)
    parser.add_argument("--skip-env-interaction", action="store_true", default=False)

    # 解析已知参数 + 收集剩余所有 "key=value" 形式的 Hydra 覆盖项
    args, unknown = parser.parse_known_args()
    hydra_overrides = [a for a in unknown if "=" in a]

    checkpoint_path = Path(args.checkpoint)
    if not (checkpoint_path / "actor").exists():
        parser.error(f"'actor' subdir not found under --checkpoint: {checkpoint_path}")

    config_file = Path(args.config)
    if not config_file.exists():
        parser.error(f"Config yaml not found: {config_file}")

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else Path.cwd() / f"analysis_results/semantic_{checkpoint_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Starting Semantic Advantage Analysis...")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  config     : {config_file}")
    print(f"  results    : {results_dir}")
    print(f"  overrides  : {hydra_overrides}")

    analysis_config = {
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "save_raw_data": True,
        "enable_semantic_eval": args.enable_semantic,
        "skip_env_interaction": args.skip_env_interaction,
        # 把 shell 里传进来的 Hydra 覆盖项原封不动地转发
        "hydra_overrides": hydra_overrides,
    }

    results = analyze_single_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_file),
        save_dir=str(results_dir),
        analysis_config=analysis_config,
    )

    print("\n🎉 Analysis done.")
    if "summary" in results:
        s = results["summary"]
        print(
            f"  mean advantage: {s.get('original_mean_advantage','N/A'):.6f} → {s.get('final_mean_advantage','N/A'):.6f}"
        )

if __name__ == "__main__":
    main()
