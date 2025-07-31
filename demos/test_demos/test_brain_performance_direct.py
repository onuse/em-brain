#!/usr/bin/env python3
"""Direct test of brain performance without server overhead."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import time
import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Direct Brain Performance Test")
print("="*60)

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,  # Standard robot sensors
    motor_dim=4,     # Standard robot motors
    spatial_resolution=32,
    quiet_mode=True
)

print("Brain initialized, warming up...")

# Warm up the brain with a few cycles
for i in range(5):
    sensory = [0.5] * 24
    brain.process_robot_cycle(sensory)

print("\nMeasuring cycle performance...")

# Measure different cycle counts
test_cycles = [10, 50, 100]

for num_cycles in test_cycles:
    cycle_times = []
    
    for i in range(num_cycles):
        # Realistic sensory input with some variation
        sensory = [0.5 + 0.1 * np.sin(i * 0.1 + j * 0.2) for j in range(24)]
        
        # Time individual cycle
        start = time.perf_counter()
        motor, state = brain.process_robot_cycle(sensory)
        cycle_time = (time.perf_counter() - start) * 1000
        
        cycle_times.append(cycle_time)
    
    # Calculate statistics
    avg_time = np.mean(cycle_times)
    std_time = np.std(cycle_times)
    min_time = np.min(cycle_times)
    max_time = np.max(cycle_times)
    
    print(f"\n{num_cycles} cycles:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Std Dev: {std_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms")
    print(f"  Max: {max_time:.1f}ms")

# Check what's taking time
print("\n\nDetailed timing breakdown (single cycle):")
sensory = [0.5] * 24

# Time each major component
timings = {}

# Field evolution
start = time.perf_counter()
brain.field_dynamics.evolve_field(brain.unified_field)
timings['field_evolution'] = (time.perf_counter() - start) * 1000

# Pattern extraction
start = time.perf_counter()
patterns = brain.pattern_system.extract_patterns(brain.unified_field, n_patterns=10)
timings['pattern_extraction'] = (time.perf_counter() - start) * 1000

# Motor generation
start = time.perf_counter()
motor_patterns = brain.motor_adapter.convert_to_motor_space(patterns)
timings['motor_generation'] = (time.perf_counter() - start) * 1000

# Total cycle
start = time.perf_counter()
motor, state = brain.process_robot_cycle(sensory)
timings['total_cycle'] = (time.perf_counter() - start) * 1000

print(f"  Field evolution: {timings['field_evolution']:.1f}ms")
print(f"  Pattern extraction: {timings['pattern_extraction']:.1f}ms")
print(f"  Motor generation: {timings['motor_generation']:.1f}ms")
print(f"  Total cycle: {timings['total_cycle']:.1f}ms")
print(f"  Other overhead: {timings['total_cycle'] - sum([v for k,v in timings.items() if k != 'total_cycle']):.1f}ms")

print(f"\n{'='*60}")
print("Summary:")
print(f"  Device: {brain.device}")
print(f"  Field shape: {brain.unified_field.shape}")
print(f"  Typical cycle time: {avg_time:.0f}ms")
print(f"  Performance: {'GOOD' if avg_time < 700 else 'DEGRADED'}")