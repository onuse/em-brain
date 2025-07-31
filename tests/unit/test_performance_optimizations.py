#!/usr/bin/env python3
"""Test that performance optimizations produce correct results."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

import torch
import time
from src.brains.field.evolved_field_dynamics import EvolvedFieldDynamics

print("Testing performance optimizations...\n")

# Test configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
field_shape = (6, 6, 6, 128)  # Small field for testing
print(f"Device: {device}")
print(f"Field shape: {field_shape}")

# Create field dynamics
dynamics = EvolvedFieldDynamics(
    field_shape=field_shape,
    device=device
)

# Create test field
field = torch.randn(field_shape, device=device) * 0.1

# Initialize dynamics
dynamics.initialize_field_dynamics(field)

# Test 1: Basic functionality
print("\n1. Testing basic functionality...")
try:
    # Run a single evolution
    evolved_field = dynamics.evolve_field(field)
    print("   ✅ Field evolution works")
    print(f"   Field shape preserved: {evolved_field.shape == field.shape}")
    print(f"   Field has changed: {not torch.allclose(evolved_field, field)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Performance comparison
print("\n2. Performance test (10 evolution cycles)...")
times = []

for i in range(10):
    start = time.perf_counter()
    evolved_field = dynamics.evolve_field(field)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    field = evolved_field  # Use evolved field for next iteration

avg_time = sum(times) / len(times)
print(f"   Average time per evolution: {avg_time*1000:.2f}ms")
print(f"   Min time: {min(times)*1000:.2f}ms")
print(f"   Max time: {max(times)*1000:.2f}ms")

# Test 3: Verify coupling and diffusion still work
print("\n3. Testing coupling and diffusion effects...")

# Create a field with a localized spike
test_field = torch.zeros(field_shape, device=device)
test_field[3, 3, 3, :64] = 1.0  # Spike in center

# Evolve several times to see diffusion
for i in range(5):
    test_field = dynamics.evolve_field(test_field)

# Check if spike has spread (diffusion working)
center_value = test_field[3, 3, 3, :64].mean().item()
neighbor_value = test_field[2, 3, 3, :64].mean().item()

print(f"   Center value after diffusion: {center_value:.4f}")
print(f"   Neighbor value after diffusion: {neighbor_value:.4f}")
print(f"   Diffusion working: {neighbor_value > 0.01}")

# Test 4: Check that the optimized code handles edge cases
print("\n4. Testing edge cases...")

# Test with minimal field
mini_field = torch.randn(2, 2, 2, 16, device=device) * 0.1
mini_dynamics = EvolvedFieldDynamics(
    field_shape=(2, 2, 2, 16),
    device=device
)
mini_dynamics.initialize_field_dynamics(mini_field)

try:
    evolved_mini = mini_dynamics.evolve_field(mini_field)
    print("   ✅ Minimal field size works")
except Exception as e:
    print(f"   ❌ Error with minimal field: {e}")

# Test with different feature sizes
try:
    large_field = torch.randn(4, 4, 4, 256, device=device) * 0.1
    large_dynamics = EvolvedFieldDynamics(
        field_shape=(4, 4, 4, 256),
        device=device
    )
    large_dynamics.initialize_field_dynamics(large_field)
    evolved_large = large_dynamics.evolve_field(large_field)
    print("   ✅ Large feature size works")
except Exception as e:
    print(f"   ❌ Error with large features: {e}")

print("\n✅ All optimization tests completed!")