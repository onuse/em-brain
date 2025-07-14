"""
Communication Subsystem (Server-Side Only)

Simple TCP server for minimal brain:
- Receives sensory vectors from any client
- Returns action vectors
- Basic capability handshake
- No client logic (clients are separate projects)
"""

from .tcp_server import MinimalTCPServer
from .protocol import MessageProtocol
from .client import MinimalBrainClient

__all__ = ["MinimalTCPServer", "MessageProtocol", "MinimalBrainClient"]