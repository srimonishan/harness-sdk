"""LlamaFirewall intervention handler."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    ToolMessage,
    UserMessage,
)

from ...hooks.events import AfterToolCallEvent, BeforeInvocationEvent, BeforeToolCallEvent
from ...interventions.actions import Confirm, Deny, LifecycleEvent, Proceed, Transform
from ...interventions.handler import InterventionHandler, OnError
from ...types.content import Messages
from ...types.tools import ToolResult

logger = logging.getLogger(__name__)


class LlamaFirewallGuard(InterventionHandler):
    """Guards an agent with LlamaFirewall scanners.

    Each lifecycle point is scanned with the scanners the firewall configures for the
    matching LlamaFirewall role:

    - ``before_invocation``: user text, as ``Role.USER``. A flagged input denies the invocation.
    - ``before_tool_call``: the tool call the model requested, as ``Role.ASSISTANT``. A blocked call
      is denied; a call that requires human review asks for confirmation.
    - ``after_tool_call``: text and JSON returned by the tool, as ``Role.TOOL``. A flagged result is
      replaced with an error result so the model never sees the flagged content.

    A role with no scanners configured is allowed without scanning. A scan that reports
    ``ScanStatus.ERROR`` raises, so ``on_error`` decides whether the agent fails, proceeds, or denies.
    With ``on_error="deny"``, a tool result that could not be scanned is replaced like a flagged one.

    Example::

        guard = LlamaFirewallGuard(firewall=LlamaFirewall(scanners={Role.USER: [ScannerType.PROMPT_GUARD]}))
        agent = Agent(interventions=[guard])
    """

    name = "llama-firewall"

    @property
    def on_error(self) -> OnError:
        """What to do when a scan fails."""
        return self._on_error

    def __init__(self, *, firewall: LlamaFirewall | None = None, on_error: OnError = "throw") -> None:
        """Initialize the LlamaFirewall guard.

        Args:
            firewall: Configured LlamaFirewall instance. Defaults to ``LlamaFirewall()``, which uses
                LlamaFirewall's default scanners for each role.
            on_error: What to do when a scan raises or reports an error status: ``'throw'`` (default)
                re-raises, ``'deny'`` denies the operation (fail-closed).

                **Warning:** ``'proceed'`` is fail-open. When a scan fails, the content passes unscanned.
        """
        self._firewall = firewall if firewall is not None else LlamaFirewall()
        self._on_error = on_error

    async def before_invocation(self, event: BeforeInvocationEvent, **kwargs: Any) -> Proceed | Deny:
        """Scan user text before the invocation begins."""
        for text in _user_texts(event.messages):
            result = await self._scan(UserMessage(content=text))
            if result.decision != ScanDecision.ALLOW:
                return Deny(reason=_describe("user input", result))
        return Proceed()

    async def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Deny | Confirm:
        """Scan the requested tool call before the tool runs."""
        tool_name = event.tool_use["name"]
        tool_input = event.tool_use.get("input")
        message = AssistantMessage(
            content=_tool_input_text(tool_input),
            tool_calls=[{"name": tool_name, "arguments": tool_input}],
        )
        result = await self._scan(message)

        if result.decision == ScanDecision.BLOCK:
            return Deny(reason=_describe(f"tool call to {tool_name}", result))
        if result.decision == ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED:
            description = _describe(f"tool call to {tool_name}", result)
            return Confirm(prompt=f"{description}. Allow this tool call?", reason=description)
        return Proceed()

    async def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any) -> Proceed | Transform:
        """Scan the tool result before the model sees it."""
        text = _tool_result_text(event.result)
        if not text:
            return Proceed()

        subject = f"result of {event.tool_use['name']}"
        try:
            result = await self._scan(ToolMessage(content=text))
        except Exception as error:
            # Deny has no effect after a tool call, so fail closed by redacting the result instead.
            if self._on_error != "deny":
                raise
            description = f"LlamaFirewall could not scan the {subject}: {error}"
            return Transform(apply=_redact_tool_result(description), reason=description)

        if result.decision == ScanDecision.ALLOW:
            return Proceed()

        description = _describe(subject, result)
        return Transform(apply=_redact_tool_result(description), reason=description)

    async def _scan(self, message: Message) -> ScanResult:
        result = await asyncio.to_thread(self._scan_blocking, message)
        logger.debug(
            "role=<%s>, decision=<%s>, score=<%s> | llamafirewall scan completed",
            message.role.value,
            result.decision.value,
            result.score,
        )
        if result.status == ScanStatus.ERROR:
            raise RuntimeError(f"LlamaFirewall scan failed: {result.reason}")
        return result

    def _scan_blocking(self, message: Message) -> ScanResult:
        # LlamaFirewall scanners run model inference synchronously, so scans run off the agent's event loop.
        return asyncio.run(self._firewall.scan_async(message))


def _describe(subject: str, result: ScanResult) -> str:
    return f"LlamaFirewall flagged the {subject} ({result.decision.value}): {result.reason}"


def _redact_tool_result(description: str) -> Callable[[LifecycleEvent], None]:
    def redact(event: LifecycleEvent) -> None:
        if not isinstance(event, AfterToolCallEvent):
            return
        event.result = ToolResult(toolUseId=event.result["toolUseId"], status="error", content=[{"text": description}])

    return redact


def _user_texts(messages: Messages | None) -> Iterator[str]:
    for message in messages or []:
        if message["role"] != "user":
            continue
        text = "\n".join(block["text"] for block in message["content"] if "text" in block)
        if text:
            yield text


def _tool_input_text(tool_input: Any) -> str:
    """Join tool input values so code passed as a string argument is scanned as written."""
    if isinstance(tool_input, str):
        return tool_input
    if isinstance(tool_input, dict):
        return "\n".join(_as_text(value) for value in tool_input.values())
    return _as_text(tool_input)


def _tool_result_text(result: ToolResult) -> str:
    parts = []
    for block in result["content"]:
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(_as_text(block["json"]))
    return "\n".join(parts)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
