from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tjipto.retrieval.router import route_retrieval


class ToolRegistryError(ValueError):
    """Raised when the server receives an invalid tool selection."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: Callable[..., Any]
    capability: str


@dataclass(frozen=True)
class ToolRegistry:
    """Server-owned names for capabilities already present in Tjipto."""

    definitions: tuple[ToolDefinition, ...]

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.definitions)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ToolRegistryError("tool_definition_invalid")
        if any(not callable(item.handler) for item in self.definitions):
            raise ToolRegistryError("tool_definition_handler_invalid")
        if len(set(names)) != len(names):
            raise ToolRegistryError("tool_definition_duplicate")

    @classmethod
    def default(cls) -> ToolRegistry:
        return cls(
            (
                ToolDefinition("retrieval_dispatch", route_retrieval, "route_retrieval"),
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions)

    def validate(self, requested: Sequence[str]) -> tuple[str, ...]:
        if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
            raise ToolRegistryError("tool_request_invalid")
        if not requested:
            raise ToolRegistryError("tool_request_empty")
        known = set(self.names)
        validated: list[str] = []
        for name in requested:
            if not isinstance(name, str) or not name.strip():
                raise ToolRegistryError("tool_name_invalid")
            if name not in known:
                raise ToolRegistryError(f"tool_not_registered:{name}")
            if name in validated:
                raise ToolRegistryError(f"tool_request_duplicate:{name}")
            validated.append(name)
        return tuple(validated)

    def require(self, name: str) -> ToolDefinition:
        self.validate((name,))
        return next(item for item in self.definitions if item.name == name)

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.require(name).handler(*args, **kwargs)


__all__ = ["ToolDefinition", "ToolRegistry", "ToolRegistryError"]
