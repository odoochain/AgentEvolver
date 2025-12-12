#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to run EvalDiplomacyWorkflow."""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentevolver.schema.task import Task
from games.games.diplomacy.workflows.eval_workflow import EvalDiplomacyWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvalDiplomacyWorkflow")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="task_config.yaml",
        help="Path to task config YAML file (default: task_config.yaml)",
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

    # 1. 读取 task_config.yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 2. 调用 workflow 运行
    task = Task(
        task_id="001",
        env_type="",
        open_query=False,
        metadata={"diplomacy_config": config_dict},
    )

    workflow = EvalDiplomacyWorkflow(
        task=task,
    )

    result = workflow.execute()
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()

