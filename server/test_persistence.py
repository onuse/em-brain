#!/usr/bin/env python3
"""
Test brain persistence to diagnose why brain_memory is empty.
"""

import sys
import os
import time
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.brains.field.pure_field_brain import PureFieldBrain
from src.persistence.integrated_persistence import IntegratedPersistence
from src.persistence.dynamic_persistence_adapter import DynamicPersistenceAdapter

print("Testing brain persistence...\n")

# Create a test brain
print("Creating PureFieldBrain...")
from src.brains.field.pure_field_brain import SCALE_CONFIGS
brain = PureFieldBrain(
    input_dim=307212,
    output_dim=6,
    scale_config=SCALE_CONFIGS['tiny'],  # Small for testing
    device='cpu'
)

print(f"Brain created:")
print(f"  Shape: {brain.field.shape}")
print(f"  Cycles: {brain.brain_cycles}")

# Run a few cycles to generate some state
print("\nRunning 5 brain cycles...")
for i in range(5):
    sensory = torch.randn(307212)
    motor = brain.process(sensory)
    print(f"  Cycle {i+1}: motor output shape = {motor.shape}")

print(f"\nBrain after processing:")
print(f"  Cycles: {brain.brain_cycles}")
print(f"  Field energy: {torch.norm(brain.field).item():.6f}")

# Test persistence
print("\n" + "=" * 60)
print("TESTING PERSISTENCE")
print("=" * 60)

memory_path = "./test_brain_memory"
print(f"Memory path: {memory_path}")

# Create persistence
persistence = IntegratedPersistence(
    memory_path=memory_path,
    save_interval_cycles=2,  # Save every 2 cycles
    auto_save=True,
    use_binary=True
)

print("\n1. Testing manual save...")
success = persistence.save_brain_state(brain, blocking=True)
if success:
    print("   ✅ Manual save successful")
else:
    print("   ❌ Manual save failed")

# Check if file was created
print("\n2. Checking saved files...")
memory_dir = Path(memory_path)
if memory_dir.exists():
    files = list(memory_dir.glob("*"))
    if files:
        print(f"   Found {len(files)} files:")
        for f in files:
            size = f.stat().st_size
            print(f"     - {f.name} ({size:,} bytes)")
    else:
        print("   ❌ No files found in memory directory")
else:
    print("   ❌ Memory directory doesn't exist")

# Test auto-save
print("\n3. Testing auto-save...")
for i in range(5):
    sensory = torch.randn(307212)
    motor = brain.process(sensory)
    # Check if auto-save triggers
    if persistence.check_auto_save(brain):
        print(f"   Cycle {brain.brain_cycles}: Auto-save triggered")
    else:
        print(f"   Cycle {brain.brain_cycles}: No auto-save")

# Final check
print("\n4. Final check of saved files...")
if memory_dir.exists():
    files = list(memory_dir.glob("*"))
    if files:
        print(f"   Found {len(files)} files after auto-save:")
        for f in files:
            size = f.stat().st_size
            print(f"     - {f.name} ({size:,} bytes)")
    else:
        print("   ❌ Still no files in memory directory")

# Get persistence stats
print("\n5. Persistence statistics:")
stats = persistence.get_persistence_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Cleanup
import shutil
if memory_dir.exists():
    shutil.rmtree(memory_dir)
    print("\nTest directory cleaned up.")