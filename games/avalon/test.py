# -*- coding: utf-8 -*-
"""Simple test script for Avalon game with API models."""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add BeyondAgent directory to path for imports
astune_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(astune_dir))

from games.avalon.game import AvalonGame


async def main(
    language: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """Main function to run Avalon game.
    
    Args:
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
        config_path: Path to config YAML file. If None, uses default config.yaml.
    """
    # Create game from config
    game = AvalonGame.from_config(
        config_path=config_path,
        language=language,
        use_user_agent=False,  # Simple test script doesn't use UserAgent
    )
    
    try:
        good_wins = await game.run()
        
        return good_wins
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Avalon game with API models")
    parser.add_argument("--config", "-c", type=str, help="Path to config YAML file")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=os.getenv("LANGUAGE", "en"),
        choices=["en", "zh", "cn", "chinese"],
        help='Language for prompts: "en" for English, "zh"/"cn"/"chinese" for Chinese (default: en)',
    )
    args = parser.parse_args()
    
    asyncio.run(
        main(
            language=args.language,
            config_path=args.config,
        )
    )
