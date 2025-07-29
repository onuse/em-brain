#!/usr/bin/env python3
"""
Blended Reality Demo

Shows how the brain seamlessly blends spontaneous dynamics (fantasy)
with sensory input (reality) based on prediction confidence.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

import time
import numpy as np
from src.core.simplified_brain_factory import SimplifiedBrainFactory


def run_blended_reality_demo():
    """
    Demonstrate blended reality behaviors:
    1. Exploration (low confidence, reality-focused)
    2. Pattern learning (building confidence)
    3. Autopilot (high confidence, fantasy-dominated)
    4. Surprise (confidence drop, reality refocus)
    5. Dreaming (no input, pure fantasy)
    """
    
    print("\n🌀 BLENDED REALITY DEMO")
    print("=" * 60)
    print("Watch how the brain blends internal simulation with sensory reality")
    print()
    
    # Create brain with blended reality
    factory = SimplifiedBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': False
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=17,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    # Helper to show blend state
    def show_blend_state(phase_name):
        if hasattr(brain, 'blended_reality'):
            state = brain.blended_reality.get_blend_state()
            print(f"\n📍 {phase_name}")
            print(f"   {state['reality_balance']}")
            print(f"   Confidence: {state['smoothed_confidence']:.2f}")
            print(f"   Dream mode: {state['dream_mode']}")
    
    # Phase 1: Exploration (Novel Environment)
    print("\n🔍 PHASE 1: EXPLORATION")
    print("-" * 40)
    print("Novel patterns → Low confidence → Reality-focused")
    
    for i in range(30):
        # Random, changing patterns
        sensors = [0.5 + 0.3 * np.sin(i * 0.3 + j) for j in range(16)] + [0.0]
        motors, state = brain.process_robot_cycle(sensors)
        
        if i % 10 == 0:
            print(f"  Cycle {i:3d}: "
                  f"Confidence={state.get('prediction_confidence', 0.5):.2f}")
    
    show_blend_state("After exploration")
    
    # Phase 2: Pattern Learning
    print("\n\n📚 PHASE 2: PATTERN LEARNING")
    print("-" * 40)
    print("Stable pattern → Building confidence → Balanced blend")
    
    # Create a stable pattern
    base_pattern = [0.3, 0.7, 0.5, 0.2] * 4 + [0.0]
    
    for i in range(50):
        motors, state = brain.process_robot_cycle(base_pattern)
        
        if i % 10 == 0:
            print(f"  Cycle {i:3d}: "
                  f"Confidence={state.get('prediction_confidence', 0.5):.2f}")
    
    show_blend_state("After learning")
    
    # Phase 3: Autopilot
    print("\n\n🚀 PHASE 3: AUTOPILOT")
    print("-" * 40)
    print("Known pattern → High confidence → Fantasy-dominated")
    
    for i in range(30):
        # Same pattern with tiny variations
        sensors = [v + 0.02 * np.sin(i * 0.1) for v in base_pattern[:-1]] + [0.5]  # Reward
        motors, state = brain.process_robot_cycle(sensors)
        
        if i % 10 == 0:
            print(f"  Cycle {i:3d}: "
                  f"Confidence={state.get('prediction_confidence', 0.5):.2f}, "
                  f"Energy={state.get('field_energy', 0.0):.4f}")
    
    show_blend_state("During autopilot")
    
    # Phase 4: Surprise
    print("\n\n⚡ PHASE 4: SURPRISE")
    print("-" * 40)
    print("Unexpected change → Confidence drop → Reality refocus")
    
    # Sudden pattern change
    surprise_pattern = [0.9, 0.1, 0.8, 0.2] * 4 + [-0.5]  # Negative reward
    
    for i in range(20):
        motors, state = brain.process_robot_cycle(surprise_pattern)
        
        if i % 5 == 0:
            print(f"  Cycle {i:3d}: "
                  f"Confidence={state.get('prediction_confidence', 0.5):.2f}")
    
    show_blend_state("After surprise")
    
    # Phase 5: Dreaming
    print("\n\n💤 PHASE 5: DREAMING")
    print("-" * 40)
    print("No input → Extended idle → Pure fantasy")
    
    # Neutral input (robot idle)
    idle_pattern = [0.5] * 16 + [0.0]
    
    for i in range(120):
        motors, state = brain.process_robot_cycle(idle_pattern)
        
        if i % 30 == 0 or (i > 90 and i % 10 == 0):
            print(f"  Cycle {i:3d}: Energy={state['field_energy']:.4f}")
            if hasattr(brain, 'blended_reality'):
                if brain.blended_reality._dream_mode:
                    print("  🌙 DREAM MODE ACTIVE - Pure spontaneous dynamics")
    
    show_blend_state("During dream")
    
    # Summary
    print("\n\n✨ SUMMARY")
    print("=" * 60)
    print("The brain seamlessly transitions between:")
    print("  - Reality-focused (exploring, learning)")
    print("  - Balanced blend (problem solving)")
    print("  - Fantasy-dominated (confident autopilot)")
    print("  - Pure fantasy (dreaming)")
    print("\nThis creates natural, adaptive behavior!")


if __name__ == "__main__":
    run_blended_reality_demo()