"""Vended intervention handlers for Strands agents.

Ready-to-use InterventionHandler implementations for common control patterns.
"""

from .hitl import HumanInTheLoop

__all__ = ["CedarAuthorization", "HumanInTheLoop", "LlamaFirewallGuard"]


def __getattr__(name: str) -> type:
    if name == "CedarAuthorization":
        from .cedar import CedarAuthorization

        return CedarAuthorization
    if name == "LlamaFirewallGuard":
        from .llama_firewall import LlamaFirewallGuard

        return LlamaFirewallGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
