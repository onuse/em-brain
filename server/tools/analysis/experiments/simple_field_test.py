#!/usr/bin/env python3
"""
Simple Field Evolution Test

Just create a field brain and call evolve_field once to see what happens.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory

def test_field_evolution():
    """Quick test of field evolution."""
    print("🧠 Simple Field Evolution Test")
    print("=" * 40)
    
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4,
            'enable_enhanced_dynamics': False,
            'enable_attention_guidance': False,
            'enable_hierarchical_processing': False
        }
    }
    
    print("Creating brain factory...")
    brain = BrainFactory(config=config, quiet_mode=True)
    
    print("Processing first sensory input...")
    test_input = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    action, brain_state = brain.process_sensory_input(test_input)
    
    print(f"First action: {[f'{x:.3f}' for x in action]}")
    print(f"Field evolution cycles: {brain_state.get('field_evolution_cycles', 'NOT_FOUND')}")
    print(f"Field energy: {brain_state.get('field_energy', 'NOT_FOUND')}")
    print(f"Prediction confidence: {brain_state.get('prediction_confidence', 'NOT_FOUND')}")
    
    print("\nProcessing second sensory input...")
    action2, brain_state2 = brain.process_sensory_input(test_input)
    
    print(f"Second action: {[f'{x:.3f}' for x in action2]}")
    print(f"Field evolution cycles: {brain_state2.get('field_evolution_cycles', 'NOT_FOUND')}")
    print(f"Field energy: {brain_state2.get('field_energy', 'NOT_FOUND')}")
    print(f"Prediction confidence: {brain_state2.get('prediction_confidence', 'NOT_FOUND')}")
    
    brain.finalize_session()
    
    return brain_state2

if __name__ == "__main__":
    test_field_evolution()