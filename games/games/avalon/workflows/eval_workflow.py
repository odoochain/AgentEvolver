# -*- coding: utf-8 -*-
"""AvalonWorkflow class for running Avalon game workflows."""
import asyncio
import os
import copy
import uuid
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict


from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from games.utils import (
    cleanup_agent_llm_clients,
    create_agent_from_config,
    create_model_from_config,
)
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from games.games.avalon.game import AvalonGame
from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment



class RoleManager:
    """Manages role indexing and identification."""
    
    def __init__(self, roles: List[tuple]):
        self.roles = roles
        self.indexed_roles = self._build_indexed_roles(roles)
    
    @staticmethod
    def _build_indexed_roles(roles: List[tuple]) -> List[str]:
        """Build indexed role names with counters."""
        role_counters = defaultdict(int)
        indexed_roles = []
        for _, role_name, _ in roles:
            indexed_roles.append(f"{role_name}_{role_counters[role_name]}")
            role_counters[role_name] += 1
        return indexed_roles
    
    def get_indexed_role(self, index: int) -> str:
        return self.indexed_roles[index]
    
    def get_role_name(self, index: int) -> str:
        return self.roles[index][1]


class EvalAvalonWorkflow:
    """Workflow class for Avalon game evaluation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize workflow with config dictionary.
        
        Args:
            config_dict: Configuration dictionary containing game settings,
                model configurations, etc.
        """
        self.config_dict = config_dict
        self.role_manager: Optional[RoleManager] = None
    
    def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """
        Get model configuration for a role.
        Role-specific config overrides default_model config.
        """
        if self.config_dict is None:
            raise ValueError("config_dict is None. Please check your configuration file.")
        
        default_model = self.config_dict.get('default_model', {})
        roles_config = self.config_dict.get('roles', {})
        
        if not isinstance(default_model, dict):
            default_model = {}
        if not isinstance(roles_config, dict):
            roles_config = {}
        
        # Start with default_model config
        config = copy.deepcopy({**default_model})
        
        # Find role-specific config (try indexed_role first, then base_role)
        role_config = next(
            (v for k, v in roles_config.items() 
             if k.lower() in [indexed_role.lower(), base_role.lower()]),
            None
        )
        
        # Override with role-specific config
        if role_config:
            config.update(role_config)
            # Handle model_name -> name mapping
            if 'model_name' in role_config:
                config['model_name'] = role_config['model_name']
        
        return config
    
    def _create_agent(self, player_id: int, indexed_role: str, base_role: str):
        """Create an agent for a player using create_agent_from_config."""
        model_config = self._get_model_config(indexed_role, base_role)
        
        # Create model using factory function
        model = create_model_from_config(model_config)
        
        # Get agent_config from model_config (should be in default_model or role-specific)
        agent_config = model_config.get('agent_config')
        if agent_config is None:
            raise ValueError(
                f"agent_config is required. Please specify it in default_model or role-specific config for {indexed_role}."
            )
        
        return create_agent_from_config(
            agent_config=agent_config,
            model=model,
            name=f"Player{player_id}",
            actor_rollout_ref=None,  # eval workflow doesn't have actor_rollout_ref
        )
    
    async def _execute_async(self) -> Dict[str, Any]:
        """Execute the game asynchronously.
        
        Returns:
            Dictionary containing game results with keys like:
            - good_victory: bool/int (1 for True, 0 for False)
            - quest_results: list of quest outcomes
            - num_quests: int (number of quests completed)
            - num_quest_successes: int (number of successful quests)
            - num_quest_failures: int (number of failed quests)
        """
        if self.config_dict is None:
            raise ValueError("config_dict is None. Please check your configuration file.")
        
        game_config = self.config_dict.get('game', {})
        if not isinstance(game_config, dict):
            game_config = {}
        
        # Setup environment and roles
        config = AvalonBasicConfig.from_num_players(game_config.get('num_players', 5))
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
        self.role_manager = RoleManager(assigned_roles)
        
        # Create agents
        self.agents = [
            self._create_agent(i, self.role_manager.get_indexed_role(i), 
                             self.role_manager.get_role_name(i))
            for i in range(len(assigned_roles))
        ]

        # Build log directory structure: logs/{experiment_name}/{timestamp}/game_id=0000
        # Get evaluation_timestamp and game_id from config (set by run_eval.py)
        # This ensures all games in the same evaluation run are organized under the same timestamp
        base_log_dir = game_config.get('log_dir', 'logs')
        evaluation_timestamp = self.config_dict.get('evaluation_timestamp')
        game_id = self.config_dict.get('game_id', 0)
        experiment_name = self.config_dict.get('experiment_name')

        for agent in self.agents:
            if game_id == 0:
                agent.set_console_output_enabled(True)
            else:
                agent.set_console_output_enabled(False)
        
        # Generate timestamp if not provided (backward compatibility)
        if not evaluation_timestamp:
            base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evaluation_timestamp = f"{base_timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Sanitize experiment_name and build directory path
        path_parts = [base_log_dir]
        if experiment_name:
            sanitized_name = str(experiment_name).replace('/', '_').replace('\\', '_')
            path_parts.append(sanitized_name)
        path_parts.append(evaluation_timestamp)
        game_timestamp = f"id={game_id:04d}"
        path_parts.append(game_timestamp)
        timestamp_dir = os.path.join(*path_parts)
        
        game = AvalonGame(
            agents=self.agents,
            config=config,
            log_dir=timestamp_dir,
            language=game_config.get('language', 'en'),
            preset_roles=assigned_roles,
        )
        
        good_victory = await game.run()
        
        # Clean up httpx client resources in agent LLM clients
        await cleanup_agent_llm_clients(self.agents)
        
        # Build result dictionary
        if good_victory is None:
            # Game was stopped
            return {
                "game_result": {
                    "good_victory": None,
                },
                "roles": [],
            }
        
        # Build roles list with indexed role_name and score
        roles_list = [
            {
                "role_name": self.role_manager.get_indexed_role(i),
                "score": 1 if (assigned_roles[i][2] == good_victory) else 0,
            }
            for i in range(len(assigned_roles))
        ]
        
        return {
            "game_result": {
                "good_victory": 1 if good_victory else 0,
            },
            "roles": roles_list,
        }
    
    def execute(self) -> Dict[str, Any]:
        """Execute the Avalon workflow.
        
        Returns:
            Dictionary containing game results.
        """
        return asyncio.run(self._execute_async())
