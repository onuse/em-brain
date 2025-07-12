"""
Experience Storage Subsystem

Stores every sensory-motor moment as a simple record:
- What I sensed (input vector)
- What I did (action vector) 
- What happened (outcome vector)
- How wrong my prediction was (error scalar)
- When this occurred (timestamp)

No categories. No types. No metadata. Just raw experience triplets.
"""

from .models import Experience
from .storage import ExperienceStorage

__all__ = ["Experience", "ExperienceStorage"]