# -*- coding: utf-8 -*-
"""Simple test script for Avalon game with API models."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add astune directory to path for imports
# test.py is at: astune/tutorial/example_avalon/test.py
# We need to add astune directory to path
astune_dir = Path(__file__).parent.parent.parent  # astune/
sys.path.insert(0, str(astune_dir))

# from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel, DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter, OpenAIMultiAgentFormatter, DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
from games.avalon.game import avalon_game
from games.avalon.engine import AvalonBasicConfig


async def main(language: str = "en"):
    """Main function to run Avalon game.
    
    Args:
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
    """
    # Configuration
    num_players = 5
    config = AvalonBasicConfig.from_num_players(num_players)
    
    # Model configuration - modify these as needed
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    api_key = os.getenv("API_KEY", "")
    base_url = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    lang_display = "中文" if language.lower() in ["zh", "cn", "chinese"] else "English"
    print(f"Initializing Avalon game with {num_players} players...")
    print(f"Language: {lang_display}")
    print(f"Model: {model_name}")
    print()
    
    # Create agents
    agents = []
    for i in range(num_players):
        # model = OpenAIChatModel(
        #     model_name=model_name,
        #     api_key=api_key,
        #     client_args={"base_url": base_url},
        #     temperature=temperature,
        # )
        model = DashScopeChatModel(
                model_name=model_name,
                api_key=api_key,
                stream=False,
                # temperature=temperature,
            )
        agent = ThinkingReActAgent(
            name=f"Player{i}",
            sys_prompt="",  # System prompt will be set in game.py
            model=model,
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )
        agents.append(agent)
        print(f"Created {agent.name}")
    
    print()
    print("=" * 60)
    print("Game Starting...")
    print("=" * 60)
    print()
    
    # Run game with logging
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        good_wins = await avalon_game(agents, config, log_dir=log_dir, language=language)
        
        print()
        print("=" * 60)
        print("Game Finished!")
        print("=" * 60)
        print(f"Result: {'Good wins!' if good_wins else 'Evil wins!'}")
        print(f"Logs saved to: {log_dir}")
        print()
        
        return good_wins
    except Exception as e:
        print(f"Error during game: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Avalon game with API models")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=os.getenv("LANGUAGE", "en"),
        choices=["en", "zh", "cn", "chinese"],
        help='Language for prompts: "en" for English, "zh"/"cn"/"chinese" for Chinese (default: en)',
    )
    args = parser.parse_args()
    
    asyncio.run(main(language=args.language))

