"""
Pure Information Stream Storage

Strategy 1: Bootstrap from Pure Information Streams

Instead of pre-structured experiences, store only raw temporal vector sequences.
Experience boundaries, input/action/outcome relationships all emerge from 
prediction patterns rather than being engineered.

This is the most radical transformation - letting the system discover what an
"experience" even means from continuous data streams.
"""

from .pure_stream_storage import PureStreamStorage
from .pattern_discovery import PatternDiscovery
from .stream_adapter import StreamToExperienceAdapter

__all__ = ["PureStreamStorage", "PatternDiscovery", "StreamToExperienceAdapter"]