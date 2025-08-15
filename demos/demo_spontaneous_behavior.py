#!/usr/bin/env python3
"""
Demo: Spontaneous Brain Behavior

This demonstrates how the brain generates autonomous behavior through
spontaneous field dynamics - the brain "thinks" even without input.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'server'))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import time


def run_spontaneous_demo():
    """Demonstrate spontaneous brain activity and autonomous behavior."""
    
    print("🧠 Spontaneous Brain Behavior Demo")
    print("=" * 60)
    print("\nThis brain has spontaneous dynamics - it 'thinks' on its own!")
    print("Watch how it generates behavior without any sensory input.\n")
    
    # Create brain with spontaneous dynamics
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': False
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=17,  # PiCar-X: 16 sensors + reward
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print("\n📊 Phase 1: Baseline Activity (no input)")
    print("-" * 60)
    print("The brain receives neutral sensory input (all 0.5)")
    print("But watch - it still generates motor commands!\n")
    
    # Neutral input - like a robot sitting still
    neutral_input = [0.5] * 16 + [0.0]  # No reward
    
    for cycle in range(20):
        motors, state = brain.process_robot_cycle(neutral_input)
        
        if cycle % 5 == 0:
            motor_str = [f"{m:+.3f}" for m in motors]
            print(f"Cycle {cycle:3d}: Motors={motor_str}, Energy={state['field_energy']:.4f}")
    
    print("\n📊 Phase 2: Brief Stimulation")
    print("-" * 60)
    print("Now giving the brain a brief 'interesting' stimulus...\n")
    
    # Interesting pattern - like seeing something novel
    interesting = [0.2, 0.8, 0.2, 0.8, 0.5, 0.5, 0.3, 0.7] + [0.5] * 8 + [0.5]  # Small reward
    
    for cycle in range(5):
        motors, state = brain.process_robot_cycle(interesting)
        motor_str = [f"{m:+.3f}" for m in motors]
        print(f"Stim {cycle}: Motors={motor_str}, Energy={state['field_energy']:.4f}")
    
    print("\n📊 Phase 3: Autonomous Behavior (post-stimulation)")
    print("-" * 60)
    print("Back to neutral input - but watch the 'afterthoughts'!\n")
    
    energy_history = []
    motor_history = []
    
    for cycle in range(30):
        motors, state = brain.process_robot_cycle(neutral_input)
        
        energy_history.append(state['field_energy'])
        motor_history.append(motors)
        
        if cycle % 5 == 0:
            motor_str = [f"{m:+.3f}" for m in motors]
            print(f"Cycle {cycle:3d}: Motors={motor_str}, Energy={state['field_energy']:.4f}")
    
    # Analyze motor activity
    motor_activity = sum(1 for motors in motor_history if any(abs(m) > 0.001 for m in motors))
    print(f"\n🎯 Motor activity: {motor_activity}/{len(motor_history)} cycles had motor output")
    
    print("\n📊 Phase 4: Different Stimulation Pattern")
    print("-" * 60)
    print("Trying a different pattern to see if brain responds differently...\n")
    
    # Different pattern
    different = [0.7, 0.7, 0.3, 0.3, 0.6, 0.4, 0.8, 0.2] + [0.5] * 8 + [0.3]
    
    for cycle in range(5):
        motors, state = brain.process_robot_cycle(different)
        motor_str = [f"{m:+.3f}" for m in motors]
        print(f"Stim {cycle}: Motors={motor_str}, Energy={state['field_energy']:.4f}")
    
    print("\n📊 Phase 5: Observing 'Personality'")
    print("-" * 60)
    print("Each brain develops its own patterns. Running longer...\n")
    
    pattern_responses = {'interesting': [], 'different': [], 'neutral': []}
    
    for _ in range(10):
        # Test each pattern
        for _ in range(5):
            _, state = brain.process_robot_cycle(interesting)
            pattern_responses['interesting'].append(state['field_energy'])
        
        for _ in range(5):
            _, state = brain.process_robot_cycle(different)
            pattern_responses['different'].append(state['field_energy'])
            
        for _ in range(5):
            _, state = brain.process_robot_cycle(neutral_input)
            pattern_responses['neutral'].append(state['field_energy'])
    
    # Calculate preferences
    avg_responses = {k: np.mean(v) for k, v in pattern_responses.items()}
    preferred = max(avg_responses, key=avg_responses.get)
    
    print("Average field response:")
    for pattern, avg in avg_responses.items():
        print(f"  {pattern}: {avg:.6f}")
    print(f"\n🎨 This brain prefers: {preferred} patterns!")
    
    print("\n✨ Key Insights:")
    print("-" * 60)
    print("1. The brain maintains activity even with neutral input")
    print("2. It generates motor commands autonomously")
    print("3. Past experiences influence future behavior")
    print("4. Each brain develops its own 'personality'")
    print("\nThis is the foundation of artificial life!")


if __name__ == "__main__":
    run_spontaneous_demo()