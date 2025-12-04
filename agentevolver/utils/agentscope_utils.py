# -*- coding: utf-8 -*-
"""Utilities for agentscope integration."""
import asyncio
import importlib
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod

from loguru import logger

from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory


def dynamic_import(module_class_str: str):
    """
    Dynamically import a class from a module.
    
    Args:
        module_class_str (str): String in format "module.path->ClassName"
        
    Returns:
        The imported class.
    """
    module_, class_ = module_class_str.split("->")
    protocol_cls = getattr(importlib.import_module(module_), class_)
    return protocol_cls


class AgentscopeModelWrapper:
    """
    A wrapper class that adapts llm_chat function to agentscope's ChatModelBase interface.
    
    This wrapper converts between agentscope's message format (role/content) and
    the internal message format (role/value), and handles async/sync conversion.
    """
    
    def __init__(
        self,
        llm_chat_fn: Callable,
        model_name: str,
        stream: bool = False,
    ):
        """
        Initialize the AgentscopeModelWrapper.
        
        Args:
            llm_chat_fn (Callable): The llm_chat function from ParallelEnvManager.
            model_name (str): The name of the model.
            stream (bool): Whether to use streaming (currently not supported, defaults to False).
        """
        # Try to import agentscope model classes
        try:
            from agentscope.model import ChatModelBase
            from agentscope.model import ChatResponse
            from agentscope.message import TextBlock
        except ImportError as e:
            logger.error(f"Failed to import agentscope model classes: {e}. "
                        "Please ensure agentscope is installed.")
            raise
        
        self.model_name = model_name
        self.stream = stream
        self.llm_chat_fn = llm_chat_fn
        
        # Store imported classes for use in methods
        self.ChatResponse = ChatResponse
        self.TextBlock = TextBlock
    
    def _convert_messages_to_internal_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert agentscope message format (role/content) to internal format (role/value).
        
        Args:
            messages (List[Dict[str, Any]]): Messages in agentscope format.
            
        Returns:
            List[Dict[str, str]]: Messages in internal format.
        """
        converted = []
        for msg in messages:
            # Extract content from various formats
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle content blocks
                text_content = ""
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif "text" in block:
                            text_content += block.get("text", "")
                    elif hasattr(block, "text"):
                        text_content += block.text
                content = text_content
            elif not isinstance(content, str):
                content = str(content) if content else ""
            
            converted.append({
                "role": msg.get("role", "user"),
                "value": content,
            })
        return converted
    
    def _convert_response_to_agentscope_format(self, response: Dict[str, str]) -> Any:
        """
        Convert internal response format to agentscope ChatResponse.
        
        Args:
            response (Dict[str, str]): Response in internal format (role/value).
            
        Returns:
            ChatResponse: Response in agentscope format.
        """
        content = response.get("value", "")
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        # Create TextBlock from content
        text_block = self.TextBlock(type="text", text=content)
        
        # Create ChatResponse
        chat_response = self.ChatResponse(
            content=[text_block],
            usage=None,  # Usage information not available from llm_chat
            metadata=None,
        )
        return chat_response
    
    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call the wrapped llm_chat function asynchronously.
        
        Args:
            messages (List[Dict[str, Any]]): Messages in agentscope format.
            tools (List[Dict] | None): Tools (not supported, ignored).
            tool_choice (str | None): Tool choice (not supported, ignored).
            **kwargs: Additional keyword arguments (e.g., custom_sampling_params, request_id).
            
        Returns:
            ChatResponse: The response in agentscope format.
        """
        # Convert messages to internal format
        internal_messages = self._convert_messages_to_internal_format(messages)
        
        # Extract custom_sampling_params and request_id from kwargs
        custom_sampling_params = kwargs.get("custom_sampling_params")
        request_id = kwargs.get("request_id")
        
        # Call llm_chat in a thread pool to make it async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm_chat_fn(
                messages=internal_messages,
                custom_sampling_params=custom_sampling_params,
                request_id=request_id,
            )
        )
        
        # Convert response to agentscope format
        return self._convert_response_to_agentscope_format(response)


class BaseAgentscopeWorkflow(ABC):
    """
    Base class for agentscope workflows.
    
    This class provides a standard interface for workflows that use agentscope agents.
    Subclasses should implement the execute() method to run the workflow and return a Trajectory.
    """
    
    def __init__(
        self,
        task: Task,
        llm_chat_fn: Callable,
        model_name: str,
        **kwargs
    ):
        """
        Initialize the workflow.
        
        Args:
            task (Task): The task to be executed.
            llm_chat_fn (Callable): The LLM chat function to use for agent creation.
            model_name (str): The name of the model.
            **kwargs: Additional keyword arguments.
        """
        self.task = task
        self.llm_chat_fn = llm_chat_fn
        self.model_name = model_name
        
        # Create agentscope model wrapper from llm_chat_fn
        self.model = AgentscopeModelWrapper(
            llm_chat_fn=llm_chat_fn,
            model_name=model_name,
            stream=False,
        )
        
        # Agents will be created by subclasses
        self.agents = []
    
    @abstractmethod
    def execute(self) -> Trajectory:
        """
        Execute the workflow and return a Trajectory.
        
        Returns:
            Trajectory: The trajectory containing model call history and workflow results.
        """
        raise NotImplementedError

