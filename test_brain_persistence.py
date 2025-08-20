#!/usr/bin/env python3
"""
Test brain persistence - save and load functionality
"""

import sys
import time
import numpy as np
sys.path.append('server/src')

from brains.field.enhanced_critical_mass_brain import EnhancedCriticalMassBrain, EmergenceConfig

print("Testing Brain Persistence")
print("=" * 60)

# Create brain with speed config
config = EmergenceConfig(
    field_size=(32, 32, 32, 64)  # Speed configuration
)

print("\n1. Creating brain and running some cycles...")
brain = EnhancedCriticalMassBrain(config)

# Run some cycles to build up state
for i in range(5):
    sensors = [50.0, 0.0, 0.0, 1.0, 25.0] + [0.0] * 7  # 12 sensor values
    motors, telemetry = brain.process(sensors)
    print(f"   Cycle {i+1}: {telemetry.get('concepts_formed', 0)} concepts, "
          f"{telemetry.get('causal_chains', 0)} chains")

print(f"\n2. Brain state before saving:")
print(f"   Cycles: {brain.metrics['cycles']}")
print(f"   Concepts: {brain.metrics['concepts_formed']}")
print(f"   Causal chains: {len(brain.predictive_chains.temporal_couplings)}")
print(f"   Semantic meanings: {len(brain.semantic_grounding.resonance_outcomes)}")

# Save the brain
save_path = "brain_states/test_persistence.brain"
print(f"\n3. Saving brain to {save_path}...")
if brain.save_state(save_path):
    print("   ✅ Save successful")
else:
    print("   ❌ Save failed")
    sys.exit(1)

# Create a new brain and load the state
print("\n4. Creating new brain and loading saved state...")
brain2 = EnhancedCriticalMassBrain(config)

print(f"   New brain before loading:")
print(f"      Cycles: {brain2.metrics['cycles']}")
print(f"      Concepts: {brain2.metrics['concepts_formed']}")

if brain2.load_state(save_path):
    print("   ✅ Load successful")
else:
    print("   ❌ Load failed")
    sys.exit(1)

print(f"\n5. Brain state after loading:")
print(f"   Cycles: {brain2.metrics['cycles']}")
print(f"   Concepts: {brain2.metrics['concepts_formed']}")
print(f"   Causal chains: {len(brain2.predictive_chains.temporal_couplings)}")
print(f"   Semantic meanings: {len(brain2.semantic_grounding.resonance_outcomes)}")

# Verify the brain can continue processing
print("\n6. Running additional cycles on loaded brain...")
for i in range(3):
    sensors = [50.0, 0.0, 0.0, 1.0, 25.0] + [0.0] * 7
    motors, telemetry = brain2.process(sensors)
    print(f"   Cycle {brain2.metrics['cycles']}: {telemetry.get('concepts_formed', 0)} concepts, "
          f"{telemetry.get('causal_chains', 0)} chains")

print("\n✅ Persistence test completed successfully!")
print("=" * 60)