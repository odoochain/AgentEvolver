# -*- coding: utf-8 -*-
"""Utility functions for the Avalon game."""
import re
from typing import Any

from agentscope.agent import AgentBase, ReActAgent
from agentscope.message import Msg

from games.avalon.prompt import EnglishPrompts as Prompts


def names_to_str(agents: list[str] | list[ReActAgent]) -> str:
    """Return a string of agent names."""
    if not agents:
        return ""
    
    names = [agent.name if isinstance(agent, ReActAgent) else agent for agent in agents]
    
    if len(names) == 1:
        return names[0]
    
    return ", ".join([*names[:-1], "and " + names[-1]])


class EchoAgent(AgentBase):
    """Echo agent that repeats the input message (Moderator for public announcements)."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Moderator"

    async def reply(self, content: str) -> Msg:
        """Repeat the input content with its name and role (public moderator announcement)."""
        msg = Msg(
            self.name,
            content,
            role="assistant",
        )
        # Print with clear label for public information
        print(f"\n[MODERATOR PUBLIC INFO] {self.name}")
        print("-" * 70)
        await self.print(msg)
        print("-" * 70 + "\n")
        return msg

    async def handle_interrupt(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Msg:
        """Handle interrupt."""

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """Observe the user's message."""


def parse_team_from_response(response: str | list) -> list[int]:
    """Parse team list from agent response."""
    response = extract_text_from_content(response)
    
    # Try to find list pattern like [0, 1, 2] or [0,1,2]
    list_match = re.search(r'\[[\s]*\d+[\s]*(?:,[\s]*\d+[\s]*)*\]', response)
    if list_match:
        return [int(n) for n in re.findall(r'\d+', list_match.group())]
    
    # Fallback: extract all numbers (limit to 10 players)
    numbers = re.findall(r'\d+', response)
    return [int(n) for n in numbers[:10]]


def parse_vote_from_response(response: str | list) -> int:
    """Parse vote (0 or 1) from agent response.
    
    Supports both English and Chinese responses:
    - Approve: yes, approve, accept, 1 (English) | 是, 批准, 同意, 通过, 赞成, 支持 (Chinese)
    - Reject: no, reject, 0 (English) | 否, 拒绝, 不同意, 失败, 反对 (Chinese)
    """
    response = extract_text_from_content(response)
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


def parse_player_id_from_response(response: str | list, max_id: int) -> int:
    """Parse player ID from agent response."""
    # Extract text from content (handles both str and list formats)
    response = extract_text_from_content(response)
    
    numbers = re.findall(r'\d+', response)
    if numbers:
        player_id = int(numbers[-1])  # Take the last number
        return max(0, min(player_id, max_id))
    return 0


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


def remove_redacted_reasoning(text: str | list) -> str:
    """
    Remove <think>...</think> or <think>...</think> tags from text.
    
    Args:
        text: Text that may contain redacted reasoning tags (can be str or list).
        
    Returns:
        Text with redacted reasoning removed.
    """
    # Extract text if content is a list
    text = extract_text_from_content(text)
    
    # Support both <think> and <think> tags
    REDACTED_PATTERN = r'<(?:redacted_reasoning|think)>.*?</(?:redacted_reasoning|think)>'
    MULTI_NEWLINE_PATTERN = r'\n\s*\n\s*\n'
    
    result = re.sub(REDACTED_PATTERN, '', text, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(MULTI_NEWLINE_PATTERN, '\n\n', result)
    return result.strip()

