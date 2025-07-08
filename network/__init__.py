"""
Network communication module for distributed brain architecture.
Provides socket-based communication between brain (laptop) and brainstem (Pi).
"""

from .brain_server import BrainSocketServer
from .brain_client import BrainSocketClient, LocalBrainClient

__all__ = [
    'BrainSocketServer',
    'BrainSocketClient', 
    'LocalBrainClient'
]