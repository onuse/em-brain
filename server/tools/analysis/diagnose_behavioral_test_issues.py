#!/usr/bin/env python3
"""
Diagnose issues with behavioral test results
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

def diagnose_prediction_confidence():
    """Diagnose why prediction confidence starts at 0.999"""
    print("🔍 Diagnosing Prediction Confidence Issue")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Test pattern
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    
    print("\n📊 Testing prediction confidence evolution:")
    print("-" * 40)
    
    # First few cycles with detailed info
    for i in range(5):
        action, brain_state = brain.process_sensory_input(pattern)
        
        print(f"\nCycle {i+1}:")
        print(f"  Prediction confidence: {brain_state.get('prediction_confidence', 0.0):.6f}")
        print(f"  Field max activation: {brain_state.get('field_max_activation', 0.0):.6f}")
        print(f"  Field mean activation: {brain_state.get('field_mean_activation', 0.0):.6f}")
        print(f"  Field total energy: {brain_state.get('field_total_energy', 0.0):.1f}")
    
    # Check internal state
    if hasattr(brain, 'unified_brain'):
        ub = brain.unified_brain
        print(f"\n🧠 Brain Internal State:")
        print(f"  Spatial resolution: {ub.spatial_resolution}³")
        print(f"  Field shape: {ub.unified_field.shape}")
        print(f"  Prediction confidence history: {ub._prediction_confidence_history[-5:]}")
        
        # Check the actual prediction error calculation
        if hasattr(ub, '_predicted_field') and ub._predicted_field is not None:
            print(f"  Predicted field exists: Yes")
            print(f"  Predicted field max: {ub._predicted_field.max().item():.6f}")
            print(f"  Predicted field mean: {ub._predicted_field.mean().item():.6f}")
        else:
            print(f"  Predicted field exists: No (first cycle)")

def diagnose_field_energy_growth():
    """Diagnose why field energy grows unboundedly"""
    print("\n\n🔍 Diagnosing Field Energy Growth")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    stable_input = [0.5] * 16
    
    print("\n📊 Testing field energy evolution:")
    print("-" * 40)
    
    energies = []
    for i in range(20):
        _, brain_state = brain.process_sensory_input(stable_input)
        energy = brain_state.get('field_total_energy', 0.0)
        energies.append(energy)
        
        if i < 5 or i % 5 == 0:
            print(f"Cycle {i+1}: energy = {energy:.1f}")
    
    # Analyze growth rate
    if len(energies) > 10:
        early_energy = np.mean(energies[:5])
        late_energy = np.mean(energies[-5:])
        growth_factor = late_energy / early_energy if early_energy > 0 else 0
        print(f"\n📈 Energy growth factor: {growth_factor:.2f}x")
        print(f"  Early average: {early_energy:.1f}")
        print(f"  Late average: {late_energy:.1f}")
    
    # Check dissipation rate
    if hasattr(brain, 'unified_brain'):
        ub = brain.unified_brain
        print(f"\n🧠 Field parameters:")
        print(f"  Energy dissipation rate: {ub.field_energy_dissipation_rate}")
        print(f"  Field evolution rate: {ub.field_evolution_rate}")
        print(f"  Maintenance cycles: {ub.maintenance_cycles}")

def analyze_test_assumptions():
    """Analyze if test assumptions match current brain behavior"""
    print("\n\n🔍 Analyzing Test Assumptions")
    print("=" * 60)
    
    print("\n📋 Current test assumptions:")
    print("1. Prediction confidence starts low (~0.5) and improves")
    print("   → Reality: Starts at 0.999 immediately")
    print("   → Issue: Prediction error calculation too sensitive or field too sparse")
    
    print("\n2. Field energy stabilizes over time")
    print("   → Reality: Energy grows exponentially")
    print("   → Issue: Dissipation rate (0.999) too high, accumulates energy")
    
    print("\n3. Pattern recognition through action differences")
    print("   → Reality: Works well (score 1.0)")
    print("   → This test is well-aligned")
    
    print("\n🔧 Suggested fixes:")
    print("1. Prediction learning:")
    print("   - Adjust prediction error sensitivity (reduce factor from 5000)")
    print("   - Ensure field has meaningful activations to predict")
    print("   - Consider measuring learning rate instead of absolute confidence")
    
    print("\n2. Field stability:")
    print("   - Reduce dissipation rate from 0.999 to ~0.95")
    print("   - Add energy normalization after each cycle")
    print("   - Implement proper energy conservation")

if __name__ == "__main__":
    diagnose_prediction_confidence()
    diagnose_field_energy_growth()
    analyze_test_assumptions()