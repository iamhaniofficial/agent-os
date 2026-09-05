"""MCPClient abstract base class and the shared SDK-session base."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any

from agentos.mcp.types import MCPServerConfig, MCPToolDef, MCPToolResult


class MCPClient(ABC):
    """Abstract base class for MCP transport clients."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def list_tools(self) -> list[MCPToolDef]:
        """List available tools from the MCP server."""

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Call a tool on the MCP server."""


class MCPSessionClient(MCPClient):
    """Shared plumbing for the two SDK-session-backed HTTP transports.

    Streamable HTTP and legacy SSE differ only in the transport they enter:
    both hold it in an ``AsyncExitStack``, talk to the server through an
    ``mcp.ClientSession``, and unwrap the same result shapes. ``connect()``
    stays with each transport; everything downstream of it lives here.
    """

    #: Used in the "not connected" error so a caller can tell the two apart.
    transport_label = "MCP"

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def close(self) -> None:
        """Close the transport, which cancels the SDK's reader task."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError(f"{self.transport_label} client is not connected")
        return self._session

    async def list_tools(self) -> list[MCPToolDef]:
        """List tools from the MCP server."""
        result = await self._require_session().list_tools()
        return [
            MCPToolDef(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Call a tool on the MCP server."""
        result = await self._require_session().call_tool(name, arguments)
        chunks: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
                continue
            if hasattr(block, "model_dump_json"):
                chunks.append(block.model_dump_json())
        structured = getattr(result, "structuredContent", None)
        if not chunks and structured is not None:
            chunks.append(json.dumps(structured, ensure_ascii=False))
        return MCPToolResult(
            content="\n".join(chunks),
            is_error=bool(getattr(result, "isError", False)),
        )
