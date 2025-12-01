# -*- coding: utf-8 -*-
"""Utility functions and classes for the Avalon game."""
import json
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
from agentscope.agent import AgentBase, ReActAgent
from loguru import logger


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Parser class for parsing agent responses."""
    
    @staticmethod
    def parse_team_from_response(response: str | list) -> list[int]:
        """Parse team list from agent response."""
        response = Parser.extract_text_from_content(response)
        
        # Try to find list pattern like [0, 1, 2] or [0,1,2]
        list_match = re.search(r'\[[\s]*\d+[\s]*(?:,[\s]*\d+[\s]*)*\]', response)
        if list_match:
            return [int(n) for n in re.findall(r'\d+', list_match.group())]
        
        # Fallback: extract all numbers (limit to 10 players)
        numbers = re.findall(r'\d+', response)
        return [int(n) for n in numbers[:10]]
    
    @staticmethod
    def parse_vote_from_response(response: str | list) -> int:
        """Parse vote (0 or 1) from agent response.
        
        Supports both English and Chinese responses:
        - Approve: yes, approve, accept, 1 (English) | 是, 批准, 同意, 通过, 赞成, 支持 (Chinese)
        - Reject: no, reject, 0 (English) | 否, 拒绝, 不同意, 失败, 反对 (Chinese)
        """
        response = Parser.extract_text_from_content(response)
        text_lower = response.lower().strip()
        text_original = response.strip()
        
        # Approve keywords: English (lowercase) + Chinese (original)
        approve_keywords = ['yes', 'approve', 'accept', '1', '是', '批准', '同意', '通过', '赞成', '支持', '一']
        # Reject keywords: English (lowercase) + Chinese (original)
        reject_keywords = ['no', 'reject', '0', '否', '拒绝', '不同意', '失败', '反对', '零']
        
        # Check both lowercase and original text
        for text in [text_lower, text_original]:
            if any(kw in text for kw in approve_keywords):
                return 1
            if any(kw in text for kw in reject_keywords):
                return 0
        
        return 0  # Default to reject
    
    @staticmethod
    def parse_player_id_from_response(response: str | list, max_id: int) -> int:
        """Parse player ID from agent response."""
        # Extract text from content (handles both str and list formats)
        response = Parser.extract_text_from_content(response)
        
        numbers = re.findall(r'\d+', response)
        if numbers:
            player_id = int(numbers[-1])  # Take the last number
            return max(0, min(player_id, max_id))
        return 0
    
    @staticmethod
    def extract_text_from_content(content: str | list) -> str:
        """Extract text string from agentscope message content (handles both str and list formats)."""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text") or item.get("content", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        return str(content)
    
    @staticmethod
    def remove_redacted_reasoning(text: str | list) -> str:
        """
        Remove <think>...</think> or <think>...</think> tags from text.
        
        Args:
            text: Text that may contain redacted reasoning tags (can be str or list).
            
        Returns:
            Text with redacted reasoning removed.
        """
        # Extract text if content is a list
        text = Parser.extract_text_from_content(text)
        
        # Support both <think> and <think> tags
        REDACTED_PATTERN = r'<(?:redacted_reasoning|think)>.*?</(?:redacted_reasoning|think)>'
        MULTI_NEWLINE_PATTERN = r'\n\s*\n\s*\n'
        
        result = re.sub(REDACTED_PATTERN, '', text, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(MULTI_NEWLINE_PATTERN, '\n\n', result)
        return result.strip()


# ============================================================================
# Logger
# ============================================================================

class GameLogger:
    """Logger class for game logging functionality."""
    
    @staticmethod
    async def save_game_logs(
        agents: list[AgentBase],
        env: Any,
        game_log: dict[str, Any],
        game_log_dir: str,
        roles: list[tuple],
    ) -> None:
        """Save game logs including agent memories and game log."""
        game_log["game_end"] = {
            "good_victory": env.good_victory,
            "quest_results": env.quest_results,
        }
        
        def convert_to_serializable(obj: Any) -> Any:
            """Convert numpy types to Python native types."""
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        game_log_data = {
            "roles": [(int(role_id), role_name, bool(side)) for role_id, role_name, side in roles],
            "game_result": {
                "good_victory": bool(env.good_victory),
                "quest_results": [bool(r) for r in env.quest_results],
            },
            "game_log": convert_to_serializable(game_log),
        }
        
        game_log_path = os.path.join(game_log_dir, "game_log.json")
        with open(game_log_path, 'w', encoding='utf-8') as f:
            json.dump(game_log_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Game log saved to {game_log_path}")
        
        # Save each agent's memory
        for i, agent in enumerate(agents):
            try:
                # Check for memory attribute (ThinkingReActAgent, TerminalUserAgent, etc.)
                if hasattr(agent, 'memory') and agent.memory is not None:
                    agent_memory = await agent.memory.get_memory()
                    memory_data = {
                        "agent_name": agent.name,
                        "agent_index": i,
                        "role": roles[i][1] if i < len(roles) else "Unknown",
                        "memory_count": len(agent_memory),
                        "memory": [msg.to_dict() for msg in agent_memory],
                    }
                    
                    memory_path = os.path.join(game_log_dir, f"{agent.name}_memory.json")
                    with open(memory_path, 'w', encoding='utf-8') as f:
                        json.dump(memory_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Agent {agent.name} memory saved to {memory_path}")
            except Exception as e:
                logger.warning(f"Failed to save memory for agent {agent.name}: {e}")
    
    @staticmethod
    def create_game_log_dir(log_dir: str | None) -> str | None:
        """Create game log directory and return the path."""
        if not log_dir:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_log_dir = os.path.join(log_dir, f"game_{timestamp}")
        os.makedirs(game_log_dir, exist_ok=True)
        logger.info(f"Game logs will be saved to: {game_log_dir}")
        return game_log_dir


# ============================================================================
# LanguageFormatter
# ============================================================================

class LanguageFormatter:
    """Language formatter helper to handle language-specific formatting."""
    
    def __init__(self, language: str = "en"):
        """Initialize language formatter with language code."""
        self.language = language
        self.is_zh = language.lower() in ["zh", "cn", "chinese"]
        self._init_localized_names()
    
    def _init_localized_names(self):
        """Initialize localized names based on language."""
        if self.is_zh:
            self.role_names = {
                "Merlin": "梅林", "Servant": "忠臣", "Assassin": "刺客", "Minion": "爪牙",
                "Percival": "派西维尔", "Morgana": "莫甘娜", "Mordred": "莫德雷德", "Oberon": "奥伯伦",
            }
            self.side_names = {"Good": "好人", "Evil": "坏人"}
            self.player_prefix = "玩家"
        else:
            self.role_names = {}
            self.side_names = {"Good": "Good", "Evil": "Evil"}
            self.player_prefix = "Player"
    
    def format_player_name(self, agent_name: str) -> str:
        """Format player name (Player0 -> 玩家 0)."""
        if self.is_zh:
            player_num = agent_name.replace("Player", "") if agent_name.startswith("Player") else agent_name
            return f"玩家 {player_num}"
        return agent_name
    
    def format_player_id(self, player_id: int) -> str:
        """Format player ID (0 -> '玩家 0' or 'Player 0')."""
        return f"{self.player_prefix} {player_id}"
    
    def format_role_name(self, role_name: str) -> str:
        """Format role name (Merlin -> 梅林)."""
        return self.role_names.get(role_name, role_name)
    
    def format_side_name(self, side: bool) -> str:
        """Format side name (True -> '好人' or 'Good')."""
        key = "Good" if side else "Evil"
        return self.side_names.get(key, key)
    
    def format_agents_names(self, agents: list[AgentBase]) -> str:
        """Format list of agent names for display."""
        if not agents:
            return ""
        
        names = [self.format_player_name(agent.name) for agent in agents]
        
        if len(names) == 1:
            return names[0]
        
        return ", ".join([*names[:-1], "和 " + names[-1] if self.is_zh else "and " + names[-1]])
    
    def format_vote_details(self, votes: list[int], approved: bool) -> tuple[str, str, str]:
        """Format vote details for display. Returns (votes_detail, result_text, outcome_text)."""
        if self.is_zh:
            votes_detail = ", ".join([f"玩家 {i}: {'批准' if v else '拒绝'}" for i, v in enumerate(votes)])
            result_text = outcome_text = "批准" if approved else "拒绝"
        else:
            votes_detail = ", ".join([f"Player {i}: {'Approve' if v else 'Reject'}" for i, v in enumerate(votes)])
            result_text = "Approved" if approved else "Rejected"
            outcome_text = "approved" if approved else "rejected"
        return votes_detail, result_text, outcome_text
    
    def format_sides_info(self, roles: list[tuple]) -> list[str]:
        """Format sides information for visibility."""
        if self.is_zh:
            return [f"玩家 {j} 是 {'好人' if s else '坏人'}" for j, (_, _, s) in enumerate(roles)]
        return [f"Player {j} is {'Good' if s else 'Evil'}" for j, (_, _, s) in enumerate(roles)]


