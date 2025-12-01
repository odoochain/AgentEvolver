# -*- coding: utf-8 -*-
"""Avalon game implemented by agentscope."""
import json
import os
from datetime import datetime
from typing import Any

from agentscope.agent import AgentBase, ReActAgent, UserAgent
from agentscope.message import Msg
from agentscope.pipeline import MsgHub, fanout_pipeline

from loguru import logger

from games.avalon.engine import AvalonGameEnvironment, AvalonBasicConfig
from games.avalon.utils import (
    EchoAgent,
    parse_team_from_response,
    parse_vote_from_response,
    parse_player_id_from_response,
)

moderator = EchoAgent()
moderator.set_console_output_enabled(True)  # Enable moderator output for public information


# ============================================================================
# Localization
# ============================================================================

class Localizer:
    """Localization helper to handle language-specific formatting."""
    
    def __init__(self, language: str = "en"):
        """Initialize localizer with language code."""
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


# ============================================================================
# Role Assignment
# ============================================================================

def _print_user_agent_private_info(agent: AgentBase, role_info: str, role_name: str, localizer: Localizer) -> None:
    """Print private information for UserAgent in terminal with clear distinction.
    
    This function outputs UserAgent's private role information separately from public moderator messages.
    """
    if isinstance(agent, UserAgent):
        print("\n" + "=" * 70)
        print(f"[USERAGENT PRIVATE INFO] {localizer.format_player_name(agent.name)} - {localizer.format_role_name(role_name)}")
        print("=" * 70)
        print(role_info)
        print("=" * 70 + "\n")


async def assign_roles_to_agents(
    agents: list[AgentBase],
    roles: list[tuple],
    Prompts: Any,
    localizer: Localizer,
) -> None:
    """Assign roles to agents and inform them of their roles and visibility."""
    MERLIN_ROLE_ID = 0
    EVIL_SIDE = 0
    
    for i, (role_id, role_name, side) in enumerate(roles):
        agent = agents[i]
        localized_role_name = localizer.format_role_name(role_name)
        side_name = localizer.format_side_name(side)
        localized_agent_name = localizer.format_player_name(agent.name)
        
        # Build visibility info
        if role_id == MERLIN_ROLE_ID or side == EVIL_SIDE:
            sides_info = localizer.format_sides_info(roles)
            additional_info = Prompts.to_agent_role_with_visibility.format(sides_info=", ".join(sides_info))
        else:
            additional_info = Prompts.to_agent_role_no_visibility
        
        role_info = Prompts.to_agent_role_assignment.format(
            agent_name=localized_agent_name,
            role_name=localized_role_name,
            side_name=side_name,
            additional_info=additional_info,
        )
        
        # Send role info to agent (private, not broadcasted)
        role_msg = Msg(
            name="Moderator",
            content=role_info,
            role="assistant",
        )
        await agent.observe(role_msg)
        
        # Print private info for UserAgent in terminal (only if UserAgent)
        _print_user_agent_private_info(agent, role_info, role_name, localizer)


# ============================================================================
# Phase Handlers
# ============================================================================

async def _handle_team_selection_phase(
    all_players_hub: MsgHub,
    agents: list[AgentBase],
    config: AvalonBasicConfig,
    env: AvalonGameEnvironment,
    mission_id: int,
    round_id: int,
    leader: int,
    game_log_dir: str | None,
    game_log: dict[str, Any],
    Prompts: Any,
    localizer: Localizer,
) -> None:
    """Handle Team Selection Phase."""
    # Broadcast phase and discussion prompt
    await all_players_hub.broadcast(
        await moderator(Prompts.to_all_team_selection_discuss.format(
            mission_id=mission_id,
            round_id=round_id,
            leader_id=leader,
            team_size=env.get_team_size(),
        ))
    )

    # Discussion: leader speaks first, then others
    leader_agent = agents[leader]
    all_players_hub.set_auto_broadcast(True)
    discussion_msgs = []
    
    # Leader speaks
    leader_msg = await leader_agent()
    discussion_msgs.append(leader_msg)
    
    # Others speak in order
    for i in range(1, config.num_players):
        agent = agents[(leader + i) % config.num_players]
        msg = await agent()
        discussion_msgs.append(msg)
    
    all_players_hub.set_auto_broadcast(False)
    discussion_msgs = [msg.to_dict() for msg in discussion_msgs]

    # Leader proposes team
    propose_prompt = await moderator(Prompts.to_leader_propose_team.format(
        mission_id=mission_id,
        team_size=env.get_team_size(),
        max_player_id=config.num_players - 1,
    ))
    team_response = await leader_agent(propose_prompt)
    team = parse_team_from_response(team_response.content)
    
    # Normalize team size
    team = list(set(team))[:env.get_team_size()]
    if len(team) < env.get_team_size():
        team.extend([i for i in range(config.num_players) if i not in team][:env.get_team_size() - len(team)])
    
    env.choose_quest_team(team=frozenset(team), leader=leader)
    
    # Record in game log
    if game_log_dir:
        game_log["missions"].append({
            "mission_id": mission_id,
            "round_id": round_id,
            "leader": leader,
            "discussion": discussion_msgs,
            "team_proposed": list(team),
        })


async def _handle_team_voting_phase(
    all_players_hub: MsgHub,
    agents: list[AgentBase],
    env: AvalonGameEnvironment,
    game_log_dir: str | None,
    game_log: dict[str, Any],
    Prompts: Any,
    localizer: Localizer,
) -> None:
    """Handle Team Voting Phase."""
    current_team = env.get_current_quest_team()
    
    # Broadcast voting phase
    vote_prompt = await moderator(Prompts.to_all_team_vote.format(team=list(current_team)))
    await all_players_hub.broadcast(vote_prompt)

    # Collect votes
    msgs_vote = await fanout_pipeline(agents, msg=vote_prompt, enable_gather=True)
    votes = [parse_vote_from_response(msg.content) for msg in msgs_vote]
    outcome = env.gather_team_votes(votes)
    
    # Format and broadcast results
    approved = bool(outcome[2])
    votes_detail, result_text, outcome_text = localizer.format_vote_details(votes, approved)
    
    result_msg = await moderator(Prompts.to_all_team_vote_result.format(
        result=result_text,
        team=list(current_team),
        outcome=outcome_text,
        votes_detail=votes_detail,
    ))
    await all_players_hub.broadcast([result_msg])
    
    # Record in game log
    if game_log_dir and game_log["missions"]:
        game_log["missions"][-1]["team_voting"] = {
            "team": list(current_team),
            "votes": votes,
            "approved": bool(outcome[2]),
        }


async def _handle_quest_voting_phase(
    all_players_hub: MsgHub,
    agents: list[AgentBase],
    env: AvalonGameEnvironment,
    mission_id: int,
    game_log_dir: str | None,
    game_log: dict[str, Any],
    Prompts: Any,
    localizer: Localizer,
) -> None:
    """Handle Quest Voting Phase."""
    current_team = env.get_current_quest_team()
    team_agents = [agents[i] for i in current_team]
    
    # Broadcast voting phase
    vote_prompt = await moderator(Prompts.to_all_quest_vote.format(team=list(current_team)))
    await all_players_hub.broadcast(vote_prompt)

    # Collect votes (private)
    msgs_vote = await fanout_pipeline(team_agents, msg=vote_prompt, enable_gather=True)
    votes = [parse_vote_from_response(msg.content) for msg in msgs_vote]
    outcome = env.gather_quest_votes(votes)
    
    # Broadcast result only
    result_msg = await moderator(Prompts.to_all_quest_result.format(
        mission_id=mission_id,
        outcome="succeeded" if outcome[2] else "failed",
        team=list(current_team),
        num_fails=outcome[3],
    ))
    await all_players_hub.broadcast(result_msg)
    
    # Record in game log
    if game_log_dir and game_log["missions"]:
        game_log["missions"][-1]["quest_voting"] = {
            "team": list(current_team),
            "votes": votes,
            "num_fails": int(outcome[3]),
            "succeeded": bool(outcome[2]),
        }


async def _handle_assassination_phase(
    all_players_hub: MsgHub,
    agents: list[AgentBase],
    env: AvalonGameEnvironment,
    config: AvalonBasicConfig,
    game_log_dir: str | None,
    game_log: dict[str, Any],
    Prompts: Any,
    localizer: Localizer,
) -> None:
    """Handle Assassination Phase."""
    # Broadcast phase
    await all_players_hub.broadcast(await moderator(Prompts.to_all_assassination))

    # Assassin chooses target
    assassin_id = env.get_assassin()
    assassinate_prompt = await moderator(
        Prompts.to_assassin_choose.format(max_player_id=config.num_players - 1)
    )
    target_response = await agents[assassin_id](assassinate_prompt)
    target = parse_player_id_from_response(target_response.content, config.num_players - 1)
    _, _, good_wins = env.choose_assassination_target(assassin_id, target)
    
    # Broadcast result
    assassin_name = localizer.format_player_id(assassin_id)
    target_name = localizer.format_player_id(target)
    result_text = Prompts.to_all_good_wins if good_wins else Prompts.to_all_evil_wins
    
    if localizer.is_zh:
        result_msg = await moderator(f"刺客{assassin_name} 选择刺杀{target_name}。{result_text}")
    else:
        result_msg = await moderator(f"Assassin {assassin_name} has chosen to assassinate {target_name}. {result_text}")
    await all_players_hub.broadcast(result_msg)
    
    # Record in game log
    if game_log_dir:
        game_log["assassination"] = {
            "assassin_id": assassin_id,
            "target": target,
            "good_wins": bool(good_wins),
        }


# ============================================================================
# Main Game Function
# ============================================================================

async def avalon_game(
    agents: list[AgentBase], 
    config: AvalonBasicConfig,
    log_dir: str | None = None,
    language: str = "en",
) -> bool:
    """The main entry of the Avalon game.

    Args:
        agents: List of agents (5-10 players). Can be ReActAgent, ThinkingReActAgent, or UserAgent.
        config: Game configuration.
        log_dir: Directory to save game logs. If None, logs are not saved.
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
    
    Returns:
        True if good wins, False otherwise.
    """
    # Initialize localizer
    localizer = Localizer(language)
    
    # Import prompts based on language
    if localizer.is_zh:
        from games.avalon.prompt import ChinesePrompts as Prompts
    else:
        from games.avalon.prompt import EnglishPrompts as Prompts
    
    assert len(agents) == config.num_players, f"The Avalon game needs exactly {config.num_players} players."

    # Disable console output for all agents (except UserAgent which handles its own output)
    # This ensures only moderator's public messages and UserAgent's private info are shown
    for agent in agents:
        if not isinstance(agent, UserAgent):
            if hasattr(agent, 'set_console_output_enabled'):
                agent.set_console_output_enabled(False)

    # Initialize logging
    game_log_dir = None
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_log_dir = os.path.join(log_dir, f"game_{timestamp}")
        os.makedirs(game_log_dir, exist_ok=True)
        logger.info(f"Game logs will be saved to: {game_log_dir}")

    # Initialize game environment
    env = AvalonGameEnvironment(config)
    roles = env.get_roles()
    
    game_log: dict[str, Any] = {
        "initialization": {
            "roles": [(role_id, role_name, side) for role_id, role_name, side in roles],
            "num_players": config.num_players,
        },
        "missions": [],
        "assassination": None,
        "game_end": None,
    }

    # Broadcast game begin message and system prompt
    async with MsgHub(participants=agents) as greeting_hub:
        # Calculate role counts for system prompt
        max_player_id = config.num_players - 1
        merlin_count = 1 if config.merlin else 0
        servant_count = config.num_good - merlin_count - (1 if config.percival else 0)
        assassin_count = 1  # Assassin always exists in standard Avalon
        minion_count = config.num_evil - assassin_count
        
        await greeting_hub.broadcast(await moderator(
            Prompts.system_prompt_template.format(
                num_players=config.num_players,
                max_player_id=max_player_id,
                num_good=config.num_good,
                merlin_count=merlin_count,
                servant_count=servant_count,
                num_evil=config.num_evil,
                assassin_count=assassin_count,
                minion_count=minion_count,
            )
        ))
        await greeting_hub.broadcast(await moderator(
            Prompts.to_all_new_game.format(localizer.format_agents_names(agents))
        ))

    # Assign roles to agents
    await assign_roles_to_agents(agents, roles, Prompts, localizer)

    # Main game loop
    while not env.done:
        phase, phase_name = env.get_phase()
        leader = env.get_quest_leader()
        mission_id = env.turn
        round_id = env.round

        async with MsgHub(participants=agents, enable_auto_broadcast=False, name="all_players") as all_players_hub:
            if phase == 0:
                await _handle_team_selection_phase(
                    all_players_hub, agents, config, env, mission_id, round_id, leader,
                    game_log_dir, game_log, Prompts, localizer
                )
            elif phase == 1:
                await _handle_team_voting_phase(
                    all_players_hub, agents, env, game_log_dir, game_log, Prompts, localizer
                )
            elif phase == 2:
                await _handle_quest_voting_phase(
                    all_players_hub, agents, env, mission_id, game_log_dir, game_log, Prompts, localizer
                )
            elif phase == 3:
                await _handle_assassination_phase(
                    all_players_hub, agents, env, config, game_log_dir, game_log, Prompts, localizer
                )

    # Game over - broadcast final result
    true_roles_str = ", ".join([
        f"{localizer.format_player_id(i)}: {localizer.format_role_name(role_name) if localizer.is_zh else role_name}"
        for i, (_, role_name, _) in enumerate(roles)
    ])
    
    async with MsgHub(participants=agents) as end_hub:
        result = Prompts.to_all_good_wins if env.good_victory else Prompts.to_all_evil_wins
        await end_hub.broadcast(await moderator(
            Prompts.to_all_game_end.format(result=result, true_roles=true_roles_str)
        ))

    logger.info(f"Game finished. Good wins: {env.good_victory}, Quest results: {env.quest_results}")
    
    # Save game log and agent memories
    if game_log_dir:
        await save_game_logs(agents, env, game_log, game_log_dir, roles)
    
    return env.good_victory


# ============================================================================
# Logging
# ============================================================================

async def save_game_logs(
    agents: list[AgentBase],
    env: AvalonGameEnvironment,
    game_log: dict[str, Any],
    game_log_dir: str,
    roles: list[tuple],
) -> None:
    """Save game logs including agent memories and game log."""
    import numpy as np
    
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
            # UserAgent may not have memory attribute
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
