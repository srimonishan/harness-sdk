"""Provide a stand-in ``llamafirewall`` module when the real package is not installed.

The real package pulls in torch and model weights, so it is not part of the test environment.
The stand-in mirrors the types the handler imports from ``llamafirewall/llamafirewall_data_types.py``.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(Enum):
    TOOL = "tool"
    USER = "user"
    ASSISTANT = "assistant"
    MEMORY = "memory"
    SYSTEM = "system"


class ScanDecision(Enum):
    ALLOW = "allow"
    HUMAN_IN_THE_LOOP_REQUIRED = "human_in_the_loop_required"
    BLOCK = "block"


class ScanStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ScanResult:
    decision: ScanDecision
    reason: str
    score: float
    status: ScanStatus = ScanStatus.SUCCESS


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[dict[str, Any]] | None = None


class UserMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role=Role.USER, content=content)


class AssistantMessage(Message):
    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        super().__init__(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)


class ToolMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role=Role.TOOL, content=content)


class LlamaFirewall:
    def __init__(self, scanners: dict[Role, list[Any]] | None = None) -> None:
        self.scanners = scanners or {}

    async def scan_async(self, input: Message, trace: list[Message] | None = None) -> ScanResult:
        return ScanResult(decision=ScanDecision.ALLOW, reason="default", score=0.0)


if importlib.util.find_spec("llamafirewall") is None:
    _module = types.ModuleType("llamafirewall")
    for _value in (
        Role,
        ScanDecision,
        ScanStatus,
        ScanResult,
        Message,
        UserMessage,
        AssistantMessage,
        ToolMessage,
        LlamaFirewall,
    ):
        setattr(_module, _value.__name__, _value)
    sys.modules["llamafirewall"] = _module
