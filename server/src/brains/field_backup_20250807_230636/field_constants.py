"""
Field Brain Constants

Shared constants used across field brain components.
"""

# Memory and topology
TOPOLOGY_REGIONS_MAX = 100  # Maximum number of memory regions before pruning

# Dunning-Kruger effect parameters
NAIVE_CONFIDENCE_BOOST_MAX = 0.5  # Maximum confidence boost for empty brain
NAIVE_MEMORY_THRESHOLD = 0.3      # Below this memory saturation, brain is "naive"
NAIVE_INTENTION_BOOST_MAX = 2.0   # Maximum intention strength multiplier for naive brain