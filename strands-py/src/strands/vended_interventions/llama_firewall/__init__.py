"""LlamaFirewall intervention handler.

Scans user input, tool calls, and tool results with Meta's LlamaFirewall to guard
an agent against prompt injection, insecure generated code, and other risks.

Requires the ``llamafirewall`` package::

    pip install strands-agents[llamafirewall]

Example::

    from llamafirewall import LlamaFirewall, Role, ScannerType
    from strands import Agent
    from strands.vended_interventions.llama_firewall import LlamaFirewallGuard

    firewall = LlamaFirewall(
        scanners={
            Role.USER: [ScannerType.PROMPT_GUARD],
            Role.ASSISTANT: [ScannerType.CODE_SHIELD],
            Role.TOOL: [ScannerType.PROMPT_GUARD],
        }
    )
    agent = Agent(interventions=[LlamaFirewallGuard(firewall=firewall)], tools=[python_repl])
"""

try:
    import llamafirewall as _llamafirewall  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "The llama_firewall intervention handler requires 'llamafirewall'. "
        "Install it with: pip install strands-agents[llamafirewall]"
    ) from _e

from .llama_firewall_guard import LlamaFirewallGuard

__all__ = [
    "LlamaFirewallGuard",
]
