# -*- coding: utf-8 -*-
"""AvalonWorkflow class for running Avalon game workflows."""
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from openai import OpenAI

from agentevolver.utils.agentscope_utils import BaseAgentscopeWorkflow
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from games.avalon.game import AvalonGame
from games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment


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


class EvalAvalonWorkflow(BaseAgentscopeWorkflow):
    """Workflow class for Avalon game evaluation."""
    
    def __init__(self, task: Task, llm_chat_fn: Any, model_name: str, **kwargs):
        super().__init__(task, llm_chat_fn, model_name, **kwargs)
        self.config_dict = task.metadata.get('avalon_config', task.metadata)
        self.role_manager: Optional[RoleManager] = None
    
    def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
        """
        Get model configuration for a role.
        Role-specific config overrides default_model config.
        """
        default_model = self.config_dict.get('default_model', {})
        roles_config = self.config_dict.get('roles', {})
        
        # Start with default_model config
        config = {**default_model}
        
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
        """Create an agent for a player."""
        from agentscope.model import OpenAIChatModel
        from agentscope.formatter import OpenAIMultiAgentFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit
        from games.avalon.agents.thinking_react_agent import ThinkingReActAgent
        
        model_config = self._get_model_config(indexed_role, base_role)
        
        # Build model kwargs
        model_kwargs = {
            'model_name': model_config['model_name'],
            'client_args': {'base_url': model_config['url']},
        }
        
        # Add optional parameters
        # Get api_key from environment variable first, then from config
        api_key = os.environ.get('API_KEY') or model_config.get('api_key')
        if api_key:
            model_kwargs['api_key'] = api_key
        if 'stream' in model_config:
            model_kwargs['stream'] = model_config['stream']
        
        # Build generate_kwargs
        generate_kwargs = {
            k: model_config[k] for k in ['temperature', 'max_tokens']
            if k in model_config
        }
        if generate_kwargs:
            model_kwargs['generate_kwargs'] = generate_kwargs
        
        model = OpenAIChatModel(**model_kwargs)
        
        return ThinkingReActAgent(
            name=f"Player{player_id}",
            sys_prompt="",
            model=model,
            formatter=OpenAIMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )
    
    async def _execute_async(self) -> Union[Trajectory, List[Trajectory]]:
        """Execute the game asynchronously."""
        game_config = self.config_dict.get('game', {})
        
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

        # Run game
        game = AvalonGame(
            agents=self.agents,
            config=config,
            log_dir=game_config.get('log_dir', 'logs'),
            language=game_config.get('language', 'en'),
            preset_roles=assigned_roles,
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        )
        
        good_victory = await game.run() or False
        return good_victory
    
    def execute(self) -> Union[Trajectory, List[Trajectory]]:
        """Execute the Avalon workflow."""
        return asyncio.run(self._execute_async())
