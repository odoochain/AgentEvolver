# -*- coding: utf-8 -*-
"""Avalon game implemented by agentscope."""
import os
from pathlib import Path
from typing import Any

import yaml

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.pipeline import MsgHub, fanout_pipeline

from loguru import logger

from games.avalon.engine import AvalonGameEnvironment, AvalonBasicConfig
from games.avalon.utils import Parser, GameLogger, LanguageFormatter
from games.avalon.agents.echo_agent import EchoAgent


class AvalonGame:
    """Main Avalon game class that integrates all game functionality."""
    
    def __init__(
        self,
        agents: list[AgentBase],
        config: AvalonBasicConfig,
        log_dir: str | None = None,
        language: str = "en",
    ):
        """Initialize Avalon game.
        
        Args:
            agents: List of agents (5-10 players). Can be ReActAgent, ThinkingReActAgent, or UserAgent.
            config: Game configuration.
            log_dir: Directory to save game logs. If None, logs are not saved.
            language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
        """
        self.agents = agents
        self.config = config
        self.log_dir = log_dir
        self.language = language
        
        # Initialize utilities
        self.localizer = LanguageFormatter(language)
        self.parser = Parser()
        self.game_logger = GameLogger()
        
        # Initialize moderator
        self.moderator = EchoAgent()
        self.moderator.set_console_output_enabled(True)
        
        # Import prompts based on language
        if self.localizer.is_zh:
            from games.avalon.prompt import ChinesePrompts as Prompts
        else:
            from games.avalon.prompt import EnglishPrompts as Prompts
        self.Prompts = Prompts
        
        # Initialize game environment
        self.env = AvalonGameEnvironment(config)
        self.roles = self.env.get_roles()
        
        # Initialize game log
        self.game_logger.create_game_log_dir(log_dir)
        self.game_logger.initialize_game_log(self.roles, config.num_players)
        
        assert len(agents) == config.num_players, f"The Avalon game needs exactly {config.num_players} players."
    
    @classmethod
    def _load_config(cls, config_path: str | None = None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def _get_config_value(cls, cfg: dict, key_path: str, env_var: str | None = None, default: Any = None) -> Any:
        """Get config value with priority: env var > config file > default."""
        if env_var and os.getenv(env_var):
            return os.getenv(env_var)
        
        keys = key_path.split('.')
        value = cfg
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value
    
    @classmethod
    def _create_agents_from_config(
        cls,
        cfg: dict,
        num_players: int,
        use_user_agent: bool,
        user_agent_id: int,
    ) -> list[AgentBase]:
        """Create agents from configuration."""
        from agentscope.model import DashScopeChatModel
        from agentscope.formatter import DashScopeMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        
        from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
        from games.avalon.agents.terminal_user_agent import TerminalUserAgent
        
        model_name = cls._get_config_value(cfg, "model.name", "MODEL_NAME", "qwen-plus")
        api_key = cls._get_config_value(cfg, "model.api_key", "API_KEY")
        stream = cfg["model"].get("stream", False)
        model_params = {k: v for k, v in cfg["model"].items() if k not in ["name", "api_key", "stream"]}
        
        def create_model(role: str | None = None) -> DashScopeChatModel:
            """Create model with optional role-specific overrides."""
            role_cfg = cfg.get("roles", {}).get(role, {}) if role else {}
            return DashScopeChatModel(
                model_name=role_cfg.get("name", model_name),
                api_key=api_key,
                stream=stream,
                **{**model_params, **{k: v for k, v in role_cfg.items() if k != "name"}}
            )
        
        agents = []
        for i in range(num_players):
            if use_user_agent and i == user_agent_id:
                agents.append(TerminalUserAgent(name=f"Player{i}"))
            else:
                agents.append(ThinkingReActAgent(
                    name=f"Player{i}",
                    sys_prompt="",
                    model=create_model(),
                    formatter=DashScopeMultiAgentFormatter(),
                    memory=InMemoryMemory(),
                    toolkit=Toolkit(),
                ))
        return agents
    
    @classmethod
    def from_config(
        cls,
        config_path: str | None = None,
        language: str | None = None,
        use_user_agent: bool | None = None,
        user_agent_id: int | None = None,
    ) -> 'AvalonGame':
        """Create AvalonGame instance from YAML configuration.
        
        Args:
            config_path: Path to config YAML file. If None, uses default config.yaml.
            language: Override language from config.
            use_user_agent: Override use_user_agent from config.
            user_agent_id: Override user_agent_id from config.
            
        Returns:
            AvalonGame instance.
        """
        cfg = cls._load_config(config_path)
        
        num_players = cfg["game"]["num_players"]
        game_language = language or cls._get_config_value(cfg, "game.language", "LANGUAGE", "en")
        log_dir = cls._get_config_value(cfg, "game.log_dir", "LOG_DIR", "logs")
        use_user_agent = use_user_agent if use_user_agent is not None else cfg["user_agent"]["enabled"]
        user_agent_id = user_agent_id if user_agent_id is not None else cfg["user_agent"]["player_id"]
        
        agents = cls._create_agents_from_config(cfg, num_players, use_user_agent, user_agent_id)
        
        os.makedirs(log_dir, exist_ok=True)
        return cls(
            agents=agents,
            config=AvalonBasicConfig.from_num_players(num_players),
            log_dir=log_dir,
            language=game_language,
        )
    
    async def run(self) -> bool:
        """Run the Avalon game.
        
        Returns:
            True if good wins, False otherwise.
        """
        # Broadcast game begin message and system prompt
        async with MsgHub(participants=self.agents) as greeting_hub:
            # Format system prompt using localizer
            system_prompt_content = self.localizer.format_system_prompt(self.config, self.Prompts)
            system_prompt_msg = await self.moderator(system_prompt_content)
            await greeting_hub.broadcast(system_prompt_msg)
            
            new_game_msg = await self.moderator(
                self.Prompts.to_all_new_game.format(self.localizer.format_agents_names(self.agents))
            )
            await greeting_hub.broadcast(new_game_msg)

        # Assign roles to agents
        await self._assign_roles_to_agents()

        # Main game loop
        while not self.env.done:
            phase, _ = self.env.get_phase()
            leader = self.env.get_quest_leader()
            mission_id = self.env.turn
            round_id = self.env.round

            async with MsgHub(participants=self.agents, enable_auto_broadcast=False, name="all_players") as all_players_hub:
                if phase == 0:
                    await self._handle_team_selection_phase(
                        all_players_hub, mission_id, round_id, leader
                    )
                elif phase == 1:
                    await self._handle_team_voting_phase(all_players_hub)
                elif phase == 2:
                    await self._handle_quest_voting_phase(all_players_hub, mission_id)
                elif phase == 3:
                    await self._handle_assassination_phase(all_players_hub)

        # Game over - broadcast final result
        async with MsgHub(participants=self.agents) as end_hub:
            end_message = self.localizer.format_game_end_message(
                self.env.good_victory,
                self.roles,
                self.Prompts
            )
            end_msg = await self.moderator(end_message)
            await end_hub.broadcast(end_msg)

        logger.info(f"Game finished. Good wins: {self.env.good_victory}, Quest results: {self.env.quest_results}")
        
        # Save game log and agent memories
        await self.game_logger.save_game_logs(self.agents, self.env, self.roles)
        
        return self.env.good_victory
    
    async def _assign_roles_to_agents(self) -> None:
        """Assign roles to agents and inform them of their roles and visibility."""
        MERLIN_ROLE_ID = 0
        EVIL_SIDE = 0
        
        for i, (role_id, role_name, side) in enumerate(self.roles):
            agent = self.agents[i]
            localized_role_name = self.localizer.format_role_name(role_name)
            side_name = self.localizer.format_side_name(side)
            localized_agent_name = self.localizer.format_player_name(agent.name)
            
            # Build visibility info
            if role_id == MERLIN_ROLE_ID or side == EVIL_SIDE:
                sides_info = self.localizer.format_sides_info(self.roles)
                additional_info = self.Prompts.to_agent_role_with_visibility.format(sides_info=", ".join(sides_info))
            else:
                additional_info = self.Prompts.to_agent_role_no_visibility
            
            role_info = self.Prompts.to_agent_role_assignment.format(
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
    
    async def _handle_team_selection_phase(
        self,
        all_players_hub: MsgHub,
        mission_id: int,
        round_id: int,
        leader: int,
    ) -> None:
        """Handle Team Selection Phase."""
        # Add mission to log
        self.game_logger.add_mission(mission_id, round_id, leader)
        
        # Broadcast phase and discussion prompt
        phase_msg = await self.moderator(self.Prompts.to_all_team_selection_discuss.format(
            mission_id=mission_id,
            round_id=round_id,
            leader_id=leader,
            team_size=self.env.get_team_size(),
        ))
        await all_players_hub.broadcast(phase_msg)

        # Discussion: leader speaks first, then others
        leader_agent = self.agents[leader]
        all_players_hub.set_auto_broadcast(True)
        discussion_msgs = []
        
        # Leader speaks
        leader_msg = await leader_agent()
        discussion_msgs.append(leader_msg)
        
        # Others speak in order
        for i in range(1, self.config.num_players):
            agent = self.agents[(leader + i) % self.config.num_players]
            msg = await agent()
            discussion_msgs.append(msg)
        
        all_players_hub.set_auto_broadcast(False)
        
        # Add discussion to log
        self.game_logger.add_discussion_messages([msg.to_dict() for msg in discussion_msgs])

        # Leader proposes team
        propose_prompt = await self.moderator(self.Prompts.to_leader_propose_team.format(
            mission_id=mission_id,
            team_size=self.env.get_team_size(),
            max_player_id=self.config.num_players - 1,
        ))
        team_response = await leader_agent(propose_prompt)
        team = self.parser.parse_team_from_response(team_response.content)
        
        # Normalize team size
        team = list(set(team))[:self.env.get_team_size()]
        if len(team) < self.env.get_team_size():
            team.extend([i for i in range(self.config.num_players) if i not in team][:self.env.get_team_size() - len(team)])
        
        self.env.choose_quest_team(team=frozenset(team), leader=leader)
        
        # Add team proposal to log
        self.game_logger.add_team_proposal(list(team))
    
    async def _handle_team_voting_phase(
        self,
        all_players_hub: MsgHub,
    ) -> None:
        """Handle Team Voting Phase."""
        current_team = self.env.get_current_quest_team()
        
        # Broadcast voting phase
        vote_prompt = await self.moderator(self.Prompts.to_all_team_vote.format(team=list(current_team)))
        await all_players_hub.broadcast(vote_prompt)

        # Collect votes
        msgs_vote = await fanout_pipeline(self.agents, msg=vote_prompt, enable_gather=True)
        votes = [self.parser.parse_vote_from_response(msg.content) for msg in msgs_vote]
        outcome = self.env.gather_team_votes(votes)
        
        # Format and broadcast results
        approved = bool(outcome[2])
        votes_detail, result_text, outcome_text = self.localizer.format_vote_details(votes, approved)
        
        result_msg = await self.moderator(self.Prompts.to_all_team_vote_result.format(
            result=result_text,
            team=list(current_team),
            outcome=outcome_text,
            votes_detail=votes_detail,
        ))
        await all_players_hub.broadcast([result_msg])
        
        # Add team voting to log
        self.game_logger.add_team_voting(list(current_team), votes, approved)
    
    async def _handle_quest_voting_phase(
        self,
        all_players_hub: MsgHub,
        mission_id: int,
    ) -> None:
        """Handle Quest Voting Phase."""
        current_team = self.env.get_current_quest_team()
        team_agents = [self.agents[i] for i in current_team]
        
        # Broadcast voting phase
        vote_prompt = await self.moderator(self.Prompts.to_all_quest_vote.format(team=list(current_team)))
        await all_players_hub.broadcast(vote_prompt)

        # Collect votes (private)
        msgs_vote = await fanout_pipeline(team_agents, msg=vote_prompt, enable_gather=True)
        votes = [self.parser.parse_vote_from_response(msg.content) for msg in msgs_vote]
        outcome = self.env.gather_quest_votes(votes)
        
        # Broadcast result only
        result_msg = await self.moderator(self.Prompts.to_all_quest_result.format(
            mission_id=mission_id,
            outcome="succeeded" if outcome[2] else "failed",
            team=list(current_team),
            num_fails=outcome[3],
        ))
        await all_players_hub.broadcast(result_msg)
        
        # Add quest voting to log
        self.game_logger.add_quest_voting(
            list(current_team),
            votes,
            int(outcome[3]),
            bool(outcome[2])
        )
    
    async def _handle_assassination_phase(
        self,
        all_players_hub: MsgHub,
    ) -> None:
        """Handle Assassination Phase."""
        # Broadcast phase
        assassination_msg = await self.moderator(self.Prompts.to_all_assassination)
        await all_players_hub.broadcast(assassination_msg)

        # Assassin chooses target
        assassin_id = self.env.get_assassin()
        assassinate_prompt = await self.moderator(
            self.Prompts.to_assassin_choose.format(max_player_id=self.config.num_players - 1)
        )
        target_response = await self.agents[assassin_id](assassinate_prompt)
        target = self.parser.parse_player_id_from_response(target_response.content, self.config.num_players - 1)
        _, _, good_wins = self.env.choose_assassination_target(assassin_id, target)
        
        # Broadcast result
        assassin_name = self.localizer.format_player_id(assassin_id)
        target_name = self.localizer.format_player_id(target)
        result_text = self.Prompts.to_all_good_wins if good_wins else self.Prompts.to_all_evil_wins
        
        if self.localizer.is_zh:
            result_msg = await self.moderator(f"刺客{assassin_name} 选择刺杀{target_name}。{result_text}")
        else:
            result_msg = await self.moderator(f"Assassin {assassin_name} has chosen to assassinate {target_name}. {result_text}")
        await all_players_hub.broadcast(result_msg)
        
        # Add assassination to log
        self.game_logger.add_assassination(assassin_id, target, bool(good_wins))
