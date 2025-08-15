#!/usr/bin/env python3
"""
Quick test to verify energy dissipation fix
"""

import sys
from pathlib import Path

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

# Create brain
config = load_adaptive_configuration("settings.json")
brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)

print("Testing energy growth with new dissipation rate (0.90)...")
print("-" * 40)

stable_input = [0.5] * 16
energies = []

for i in range(150):  # Run longer to see if it stabilizes
    _, brain_state = brain.process_sensory_input(stable_input)
    energy = brain_state.get('field_total_energy', 0.0)
    energies.append(energy)
    
    if i < 5 or i % 25 == 0:
        print(f"Cycle {i}: energy = {energy:.1f}")

# Check if energy stabilized
if len(energies) > 100:
    early_avg = sum(energies[20:40]) / 20
    late_avg = sum(energies[-20:]) / 20
    growth = (late_avg - early_avg) / early_avg * 100
    
    print(f"\nEnergy growth: {growth:.1f}%")
    print(f"Early average: {early_avg:.1f}")
    print(f"Late average: {late_avg:.1f}")
    
    if abs(growth) < 10:
        print("✅ Energy appears to be stabilizing!")
    else:
        print("⚠️  Energy still growing significantly")