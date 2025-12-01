# -*- coding: utf-8 -*-
"""Example test script with role-specific model configuration and UserAgent support."""
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add BeyondAgent directory to path for imports
ba_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ba_dir))

from games.avalon.game import AvalonGame


async def main(
    language: Optional[str] = None,
    use_user_agent: Optional[bool] = None,
    user_agent_id: Optional[int] = None,
    config_path: Optional[str] = None,
):
    """Main function to run Avalon game with role-specific configurations."""
    # Create game from config
    game = AvalonGame.from_config(
        config_path=config_path,
        language=language,
        use_user_agent=use_user_agent,
        user_agent_id=user_agent_id,
    )
    
    try:
        good_wins = await game.run()
        return good_wins
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Avalon game with role-specific configurations")
    parser.add_argument("--config", "-c", type=str, help="Path to config YAML file")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        choices=["en", "zh", "cn", "chinese"],
        help='Language: "en" for English, "zh"/"cn"/"chinese" for Chinese',
    )
    parser.add_argument("--use-user-agent", action="store_true", help="Enable UserAgent")
    parser.add_argument("--no-user-agent", action="store_true", help="Disable UserAgent")
    parser.add_argument("--user-agent-id", type=int, help="Player ID for UserAgent (0-indexed)")
    args = parser.parse_args()

    asyncio.run(
        main(
            language=args.language,
            use_user_agent=True if args.use_user_agent else (False if args.no_user_agent else None),
            user_agent_id=args.user_agent_id,
            config_path=args.config,
        )
    )
