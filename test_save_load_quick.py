#!/usr/bin/env python3
"""
Quick test of save/load functionality
"""

import sys
import os
sys.path.append('server/src')

from brains.field.enhanced_critical_mass_brain import EnhancedCriticalMassBrain, EmergenceConfig

print("Quick Save/Load Test")
print("=" * 40)

# Create minimal brain
config = EmergenceConfig(
    field_size=(8, 8, 8, 16)  # Tiny for quick test
)

brain = EnhancedCriticalMassBrain(config)

# Set some test values
brain.metrics['cycles'] = 100
brain.metrics['concepts_formed'] = 5
brain.predictive_chains.temporal_couplings = {(1, 2): 0.5, (2, 3): 0.7}
brain.semantic_grounding.resonance_outcomes = {1: [], 2: []}

print(f"Before save: cycles={brain.metrics['cycles']}, concepts={brain.metrics['concepts_formed']}")

# Save
os.makedirs("brain_states", exist_ok=True)
if brain.save_state("brain_states/quick_test.brain"):
    print("✅ Save successful")
else:
    print("❌ Save failed")
    sys.exit(1)

# Create new brain and load
brain2 = EnhancedCriticalMassBrain(config)
if brain2.load_state("brain_states/quick_test.brain"):
    print("✅ Load successful")
else:
    print("❌ Load failed")
    sys.exit(1)

print(f"After load: cycles={brain2.metrics['cycles']}, concepts={brain2.metrics['concepts_formed']}")
print(f"Causal chains: {len(brain2.predictive_chains.temporal_couplings)}")
print(f"Semantic meanings: {len(brain2.semantic_grounding.resonance_outcomes)}")

if brain2.metrics['cycles'] == 100 and brain2.metrics['concepts_formed'] == 5:
    print("\n✅ Persistence working correctly!")
else:
    print("\n❌ Values don't match!")