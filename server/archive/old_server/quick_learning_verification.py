#!/usr/bin/env python3
"""
Quick Learning Verification
Test what energy trends look like in a successful learning scenario
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_factory import BrainFactory

def test_energy_trends():
    """Test what energy trends look like during learning"""
    print("🔍 Testing field energy trends during learning...")
    
    # Clear memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
    
    energies = []
    for i in range(20):
        action, brain_state = brain.process_sensory_input(pattern)
        field_energy = brain_state.get('field_energy', 0.0)
        energies.append(field_energy)
        
        if i % 5 == 0:
            print(f"   Cycle {i}: Field energy = {field_energy:.6f}")
    
    # Calculate trends like the field logger
    if len(energies) >= 10:
        recent_energies = energies[-10:]
        early = np.mean(recent_energies[:3])
        recent_mean = np.mean(recent_energies[-3:])
        energy_trend = recent_mean - early
        
        print(f"\n📊 Energy Trend Analysis:")
        print(f"   Early average: {early:.6f}")
        print(f"   Recent average: {recent_mean:.6f}")
        print(f"   Energy trend: {energy_trend:+.6f}")
        print(f"   Learning detected (old): {energy_trend > 0}")
        print(f"   Learning detected (new): {energy_trend > -0.001}")
    
    brain.finalize_session()

if __name__ == "__main__":
    test_energy_trends()