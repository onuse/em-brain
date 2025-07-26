"""
Communication Subsystem (Server-Side Only)

Simple TCP server for minimal brain:
- Receives sensory vectors from any client
- Returns action vectors
- Basic capability handshake
- No client logic (clients are separate projects)
"""

# Old tcp_server moved to archive, use clean_tcp_server instead
from .clean_tcp_server import CleanTCPServer
from .protocol import MessageProtocol
from .client import MinimalBrainClient

__all__ = ["CleanTCPServer", "MessageProtocol", "MinimalBrainClient"]