#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to run EvalAvalonWorkflow."""
import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentevolver.schema.task import Task
from games.avalon.workflows.eval_workflow import EvalAvalonWorkflow


def main():
    parser = argparse.ArgumentParser(description="Run EvalAvalonWorkflow")
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
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 2. 调用 workflow 运行
    task = Task(
        task_id="eval_task",
        env_type="avalon",
        open_query=False,
        metadata={"avalon_config": config_dict},
    )
    
    def llm_chat(messages, custom_sampling_params=None, request_id=None):
        return {"role": "assistant", "value": ""}
    
    workflow = EvalAvalonWorkflow(
        task=task,
        llm_chat_fn=llm_chat,
        model_name="eval-model",
    )
    
    workflow.execute()


if __name__ == "__main__":
    main()

