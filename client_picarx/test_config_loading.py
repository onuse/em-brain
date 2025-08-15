#!/usr/bin/env python3
"""
Test that configuration is loaded correctly from robot_config.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from brainstem.brainstem import load_robot_config

print("Testing configuration loading...\n")

# Load config
config = load_robot_config()

# Show brain config
brain_config = config.get("brain", {})
print("Brain configuration from robot_config.json:")
print(f"  Host: {brain_config.get('host', 'NOT SET')}")
print(f"  Port: {brain_config.get('port', 'NOT SET')}")
print(f"  Timeout: {brain_config.get('timeout', 'NOT SET')}")

print("\nTesting Brainstem initialization with no arguments...")
from brainstem.brainstem import Brainstem

# Create brainstem with no args - should use config
brainstem = Brainstem(enable_brain=False)  # Don't actually connect
print(f"\nBrainstem brain configuration:")
print(f"  Host: {brainstem.brain_host}")
print(f"  Port: {brainstem.brain_port}")

# Test with override
print("\nTesting with override to 10.0.0.1...")
brainstem2 = Brainstem(brain_host="10.0.0.1", enable_brain=False)
print(f"  Host: {brainstem2.brain_host}")
print(f"  Port: {brainstem2.brain_port}")

print("\n✅ Configuration test complete")