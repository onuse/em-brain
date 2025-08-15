#!/usr/bin/env python3
"""
Test if our prediction timing and energy normalization fixes work
"""

import sys
from pathlib import Path
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

def test_prediction_fix():
    """Test if prediction confidence now shows meaningful learning"""
    print("🔬 Testing Prediction Confidence Fix")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    confidences = []
    
    print("\nPrediction confidence evolution:")
    print("-" * 40)
    
    for i in range(20):
        _, brain_state = brain.process_sensory_input(pattern)
        conf = brain_state.get('prediction_confidence', 0.0)
        confidences.append(conf)
        
        if i < 10 or i % 5 == 0:
            print(f"Cycle {i+1}: confidence = {conf:.6f}")
    
    # Check if we see learning
    early_conf = np.mean(confidences[:5])
    late_conf = np.mean(confidences[-5:])
    improvement = late_conf - early_conf
    
    print(f"\nEarly average: {early_conf:.6f}")
    print(f"Late average: {late_conf:.6f}")
    print(f"Improvement: {improvement:.6f}")
    
    if early_conf < 0.9 and improvement > 0.05:
        print("✅ Prediction learning appears to be working!")
    else:
        print("⚠️  Prediction learning still needs work")

def test_energy_normalization():
    """Test if energy normalization prevents unbounded growth"""
    print("\n\n🔬 Testing Energy Normalization")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    stable_input = [0.5] * 16
    energies = []
    
    print("\nField energy evolution:")
    print("-" * 40)
    
    for i in range(200):  # Longer test to see normalization kick in
        _, brain_state = brain.process_sensory_input(stable_input)
        energy = brain_state.get('field_total_energy', 0.0)
        energies.append(energy)
        
        if i < 5 or i % 25 == 0:
            print(f"Cycle {i}: energy = {energy:.0f}")
    
    # Check if energy stabilized
    max_energy = max(energies)
    final_energy = energies[-1]
    
    print(f"\nMax energy reached: {max_energy:.0f}")
    print(f"Final energy: {final_energy:.0f}")
    
    if max_energy < 200000 and final_energy < 150000:
        print("✅ Energy normalization is working!")
    else:
        print("⚠️  Energy still growing too much")

def test_behavioral_improvement():
    """Quick behavioral test to see if scores improved"""
    print("\n\n🔬 Quick Behavioral Assessment")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Test 1: Prediction learning (30 seconds)
    print("\n📊 Prediction Learning Test:")
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    confidences = []
    
    for i in range(100):
        _, brain_state = brain.process_sensory_input(pattern)
        confidences.append(brain_state.get('prediction_confidence', 0.0))
    
    if len(confidences) > 20:
        early = np.mean(confidences[:10])
        late = np.mean(confidences[-10:])
        score = min(1.0, max(0.0, (late - early) * 2))
        print(f"Score: {score:.3f} (target: >0.3)")
    
    # Test 2: Field stability (30 seconds)
    print("\n📊 Field Stability Test:")
    stable_input = [0.5] * 16
    energies = []
    
    for i in range(100):
        _, brain_state = brain.process_sensory_input(stable_input)
        energies.append(brain_state.get('field_total_energy', 0.0))
    
    if len(energies) > 50:
        early_var = np.var(energies[:25])
        late_var = np.var(energies[-25:])
        stability_improvement = max(0, (early_var - late_var) / max(early_var, 1))
        score = min(1.0, stability_improvement)
        print(f"Score: {score:.3f} (target: >0.1)")

if __name__ == "__main__":
    test_prediction_fix()
    test_energy_normalization()
    test_behavioral_improvement()