#!/usr/bin/env python3
"""Test single memory function."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tests.integration.test_memory_prediction import MemoryPredictionTester

# Create tester
tester = MemoryPredictionTester(spatial_resolution=8, quiet_mode=False)

# Run a simple test
print("Running simple action test...")

# Create a simple pattern
pattern = [0.9] * 24
action, state = tester.brain.process_robot_cycle(pattern)

print(f"\nAction output: {action}")
print(f"Action type: {type(action)}")
print(f"Action length: {len(action) if hasattr(action, '__len__') else 'N/A'}")

if len(action) > 0:
    print(f"First few values: {action[:5]}")
    print(f"Contains NaN: {any(np.isnan(action))}")