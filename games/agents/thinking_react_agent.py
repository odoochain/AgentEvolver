"""ReAct agent with private thinking (shared)."""

from typing import Any, Literal
import re

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from games.agents.utils import extract_text_from_content


class ThinkingReActAgent(ReActAgent):
    """A ReAct agent that thinks before speaking."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter,
        toolkit=None,
        memory=None,
        long_term_memory=None,
        long_term_memory_mode: Literal["agent_control", "static_control", "both"] = "both",
        enable_meta_tool: bool = False,
        parallel_tool_calls: bool = False,
        knowledge=None,
        enable_rewrite_query: bool = True,
        plan_notebook=None,
        print_hint_msg: bool = False,
        max_iters: int = 10,
        thinking_sys_prompt: str | None = None,
    ) -> None:
        """Initialize a ThinkingReActAgent."""
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=memory,
            long_term_memory=long_term_memory,
            long_term_memory_mode=long_term_memory_mode,
            enable_meta_tool=enable_meta_tool,
            parallel_tool_calls=parallel_tool_calls,
            knowledge=knowledge,
            enable_rewrite_query=enable_rewrite_query,
            plan_notebook=plan_notebook,
            print_hint_msg=print_hint_msg,
            max_iters=max_iters,
        )

        if thinking_sys_prompt is None:
            thinking_sys_prompt = (
                "Before you respond, think carefully about your response. "
                "Your thinking process should be wrapped in <think>...</think> tags. "
                "Then provide your actual response after the thinking section. "
                "Example format:\n"
                "<think>\n"
                "Your private thinking here...\n"
                "</think>\n"
                "Your actual response here."
            )

        self._sys_prompt = f"{self._sys_prompt}\n\n{thinking_sys_prompt}"
        self.model_call_history: list[dict[str, Any]] = []

    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "any", "required"] | None = None,
    ) -> Msg:
        """Perform reasoning with thinking section.

        The complete message (with thinking) is stored in memory/history,
        but the returned message (for broadcast) excludes thinking content.
        """
        prompt = await self.formatter.format(
            msgs=[
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
                *await self._reasoning_hint_msgs.get_memory(),
            ],
        )

        msg = await super()._reasoning(tool_choice)

        if msg is not None:
            response_content = extract_text_from_content(msg.content)
            call_record = {
                "prompt": prompt,  # prompt is already list[dict[str, Any]]
                "response": response_content,
                "response_msg": msg.to_dict()
                if hasattr(msg, "to_dict")
                else {
                    "name": msg.name,
                    "content": response_content,
                    "role": msg.role,
                    "timestamp": str(msg.timestamp) if hasattr(msg, "timestamp") else None,
                },
            }
            self.model_call_history.append(call_record)

        if msg is None:
            return msg

        _, public_msg = self._separate_thinking_and_response(msg)

        return_msg = Msg(
            name=msg.name,
            content=public_msg.content,
            role=msg.role,
            metadata=msg.metadata,
        )
        return_msg.id = msg.id
        return_msg.timestamp = msg.timestamp

        return return_msg

    def _separate_thinking_and_response(
        self,
        msg: Msg,
    ) -> tuple[Msg | None, Msg]:
        """Separate thinking content from public response."""
        # Prefer Msg.get_text_content() when available, fallback to robust extractor.
        text_content = (
            msg.get_text_content()
            if hasattr(msg, "get_text_content")
            else extract_text_from_content(msg.content)
        )

        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, text_content or "", re.DOTALL)

        thinking_content: str | None = None
        if matches:
            thinking_content = matches[0].strip()
            public_text = re.sub(pattern, "", text_content, flags=re.DOTALL).strip()
        else:
            public_text = (text_content or "").strip()

        thinking_msg: Msg | None = None
        if thinking_content:
            thinking_msg = Msg(
                name=self.name,
                content=[
                    TextBlock(
                        type="text",
                        text=f"<think>\n{thinking_content}\n</think>",
                    ),
                ],
                role="assistant",
            )

        public_blocks: list[Any] = []

        def _get_block_type_and_text(block: Any) -> tuple[str | None, str | None]:
            if isinstance(block, dict):
                return block.get("type"), block.get("text")
            return getattr(block, "type", None), getattr(block, "text", None)

        if isinstance(msg.content, str):
            if public_text:
                public_blocks = [TextBlock(type="text", text=public_text)]
            final_content: Any = public_blocks if public_blocks else ""
        elif isinstance(msg.content, list):
            for block in msg.content:
                block_type, block_text = _get_block_type_and_text(block)

                # Skip explicit "thinking" blocks if present.
                if block_type == "thinking":
                    continue

                if block_type == "text":
                    cleaned_text = re.sub(
                        pattern,
                        "",
                        (block_text or ""),
                        flags=re.DOTALL,
                    ).strip()
                    if cleaned_text:
                        public_blocks.append(TextBlock(type="text", text=cleaned_text))
                else:
                    # Keep other block types (tool_use, image, audio, etc.)
                    public_blocks.append(block)

            final_content = public_blocks if public_blocks else []
        else:
            # Unknown content type; fall back to public_text.
            final_content = [TextBlock(type="text", text=public_text)] if public_text else ""

        public_msg = Msg(
            name=msg.name,
            content=final_content,
            role=msg.role,
            metadata=msg.metadata,
        )
        public_msg.id = msg.id
        public_msg.timestamp = msg.timestamp

        return thinking_msg, public_msg
