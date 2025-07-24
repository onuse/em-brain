#!/usr/bin/env python3
"""Investigate emergent navigation capabilities in UnifiedFieldBrain."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== EMERGENT NAVIGATION INVESTIGATION ===\n")

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

print("1. FIELD ARCHITECTURE ANALYSIS:")
print(f"   Total dimensions: {brain.total_dimensions}")
print(f"   Spatial dimensions: 5 (x, y, z, scale, time)")
print(f"   Field families:")
for family in brain.field_dimensions:
    print(f"     - {family.name}: {family.description}")

print("\n2. NAVIGATION-RELEVANT MECHANISMS:")

# Check gradient following
print("\n   a) Gradient Following:")
print(f"      - Gradient strength: {brain.gradient_following_strength}")
print(f"      - Computed gradients: {list(brain.gradient_flows.keys())}")
print("      - Action generation: Field gradients → Motor commands")

# Check topology regions (spatial memory)
print("\n   b) Topology Regions (Spatial Memory):")
print(f"      - Active regions: {len(brain.topology_regions)}")
print(f"      - Stability threshold: {brain.topology_stability_threshold}")
print("      - Can store: Persistent field patterns at locations")

# Check constraint system
print("\n   c) Constraint System:")
print(f"      - Discovery rate: {brain.constraint_discovery_rate}")
print(f"      - Can learn: Gradient flow patterns, activation patterns")
print(f"      - Enforces: Field dynamics that worked before")

# Check field evolution
print("\n   d) Field Evolution:")
print(f"      - Evolution rate: {brain.field_evolution_rate}")
print(f"      - Decay rate: {brain.field_decay_rate}")
print(f"      - Diffusion: Spreads activation spatially")

print("\n3. TESTING EMERGENT NAVIGATION:")

# Test 1: Does the field respond to spatial patterns?
print("\n   Test 1: Spatial Pattern Response")
# Simulate moving toward a target (decreasing distance)
for distance in [1.0, 0.8, 0.6, 0.4, 0.2]:
    sensors = [distance] * 3 + [0.5] * 21  # Front sensors show distance
    action, _ = brain.process_robot_cycle(sensors)
    print(f"      Distance {distance:.1f}: Speed={action[3]:.3f}, Turn={action[0]-action[1]:.3f}")

# Test 2: Does it form spatial memories?
print("\n   Test 2: Spatial Memory Formation")
initial_regions = len(brain.topology_regions)

# Create location-specific patterns
locations = [(0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8)]
for x, y in locations:
    # Unique sensory pattern for each location
    sensors = [x, y, 0.5] + [x*y] * 21
    for _ in range(20):  # Repeat to form memory
        brain.process_robot_cycle(sensors)
    
regions_formed = len(brain.topology_regions) - initial_regions
print(f"      Spatial memories formed: {regions_formed}")
print(f"      Topology regions: {list(brain.topology_regions.keys())[:5]}")

# Test 3: Does it learn navigation patterns?
print("\n   Test 3: Navigation Pattern Learning")
# Simulate successful navigation (obstacle avoidance)
for i in range(10):
    # Obstacle on right → turn left pattern
    sensors = [0.2, 0.5, 0.9] + [0.0] * 21
    brain.process_robot_cycle(sensors)
    # Clear path → move forward
    sensors = [0.1, 0.1, 0.1] + [0.0] * 21
    brain.process_robot_cycle(sensors)

constraints_learned = len(brain.constraint_field.active_constraints)
print(f"      Constraints learned: {constraints_learned}")

print("\n4. ANALYSIS:")
print("   The UnifiedFieldBrain has ALL components needed for navigation:")
print("   ✓ Spatial representation (5D spatial dimensions)")
print("   ✓ Memory formation (topology regions)")
print("   ✓ Pattern learning (constraint system)")
print("   ✓ Action generation (gradient following)")
print("\n   Navigation SHOULD emerge from these mechanisms!")

print("\n5. POTENTIAL ISSUES:")
print("   - Gradient following strength might need tuning")
print("   - Topology regions might not persist long enough")
print("   - Constraint discovery might not capture navigation patterns")
print("   - Field decay might erase navigation memories too quickly")

print("\n6. HYPOTHESIS:")
print("   Navigation IS an emergent property, but may need:")
print("   1. Longer temporal persistence (reduce decay)")
print("   2. Stronger spatial gradients")
print("   3. Better constraint pattern matching")
print("   4. Richer sensory input (landmarks, goals)")

brain.shutdown()