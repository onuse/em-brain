#!/usr/bin/env python3
"""
Test if persistence serialization error is fixed
"""

import sys
from pathlib import Path

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

print("Testing persistence fix...")
print("-" * 40)

# Create brain with persistence enabled
config = load_adaptive_configuration("settings.json")
brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)

# Run for 110 cycles to trigger incremental save (happens every 100 cycles)
pattern = [0.5] * 16
for i in range(110):
    brain.process_sensory_input(pattern)
    if i % 20 == 0:
        print(f"Cycle {i}")

print("\n✅ Test completed - check for persistence errors above")