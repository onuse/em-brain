#!/usr/bin/env python3
"""Test that server shutdown doesn't crash with persistence error."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.core.brain_service import BrainService
from src.core.brain_pool import BrainPool
from src.core.simplified_brain_factory import SimplifiedBrainFactory
from src.core.simplified_adapters import SimplifiedAdapterFactory

print("Testing server shutdown...")

# Create minimal components
brain_factory = SimplifiedBrainFactory(brain_config={})
brain_pool = BrainPool(brain_factory=brain_factory)
adapter_factory = SimplifiedAdapterFactory()

# Create brain service
brain_service = BrainService(
    brain_pool=brain_pool,
    adapter_factory=adapter_factory,
    quiet=True
)

# Test shutdown
print("Calling shutdown...")
try:
    brain_service.shutdown()
    print("✅ Shutdown completed without errors!")
except AttributeError as e:
    if "persistence" in str(e):
        print(f"❌ Persistence error during shutdown: {e}")
    else:
        raise
except Exception as e:
    print(f"❌ Unexpected error during shutdown: {e}")
    raise