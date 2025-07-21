#!/usr/bin/env python3
"""
Test Prediction-Driven Learning

Verify that the brain now uses prediction accuracy to modulate learning rates
and generates exploration actions instead of staying still.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory
import time

def test_prediction_driven_learning():
    """
    Test that the brain uses prediction accuracy to drive learning.
    """
    print("🧠 Prediction-Driven Learning Test")
    print("=" * 40)
    
    # Create field brain
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4,
            'enable_enhanced_dynamics': True,
            'enable_attention_guidance': True,
            'enable_hierarchical_processing': True
        }
    }
    
    print("Creating field brain...")
    brain = BrainFactory(config=config, quiet_mode=False)
    
    print(f"\n🔬 Testing prediction-driven learning mechanics:")
    
    # Test 1: Check that actions are not all zeros (exploration fix)
    print(f"\n📊 Test 1: Action Generation (Anti-Stillness)")
    actions_generated = []
    for i in range(10):
        sensory_input = [0.1 + i * 0.05] * 8
        action, brain_state = brain.process_sensory_input(sensory_input)
        actions_generated.append(action)
        action_magnitude = sum(abs(x) for x in action)
        print(f"   Cycle {i+1}: action_magnitude={action_magnitude:.4f}, actions={[f'{x:.3f}' for x in action]}")
    
    # Check if we're generating non-zero actions
    non_zero_actions = sum(1 for actions in actions_generated if sum(abs(x) for x in actions) > 0.01)
    print(f"   Non-zero actions: {non_zero_actions}/10")
    
    # Test 2: Check prediction-driven learning rate adaptation
    print(f"\n🎯 Test 2: Prediction-Driven Learning Rate Adaptation")
    
    # Process more inputs to build prediction history
    for i in range(15):
        sensory_input = [0.2 + i * 0.03] * 8
        action, brain_state = brain.process_sensory_input(sensory_input)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        evolution_cycles = brain_state.get('field_evolution_cycles', 0)
        
        if i % 5 == 4:  # Report every 5 cycles
            print(f"   Cycle {i+1}: confidence={confidence:.3f}, evolution_cycles={evolution_cycles}")
    
    # Get brain internals to check prediction history
    field_brain = brain.brain_adapter.field_brain
    prediction_history = getattr(field_brain, 'prediction_accuracy_history', [])
    current_evolution_rate = getattr(field_brain, 'field_evolution_rate', 0.1)
    base_evolution_rate = getattr(field_brain, 'base_field_evolution_rate', 0.1)
    
    print(f"   Prediction history length: {len(prediction_history)}")
    if prediction_history:
        recent_accuracy = sum(prediction_history[-3:]) / min(len(prediction_history), 3)
        print(f"   Recent prediction accuracy: {recent_accuracy:.3f}")
        print(f"   Base evolution rate: {base_evolution_rate:.3f}")
        print(f"   Current evolution rate: {current_evolution_rate:.3f}")
        
        if current_evolution_rate != base_evolution_rate:
            print(f"   ✅ Prediction-driven rate adaptation working!")
        else:
            print(f"   ⚠️ Rate adaptation not yet triggered")
    
    # Test 3: Verify field evolution cycles are incrementing
    print(f"\n🔄 Test 3: Field Evolution Progression")
    initial_cycles = brain_state.get('field_evolution_cycles', 0)
    
    # Process a few more inputs
    for i in range(5):
        sensory_input = [0.3 + i * 0.02] * 8
        action, brain_state = brain.process_sensory_input(sensory_input)
    
    final_cycles = brain_state.get('field_evolution_cycles', 0)
    cycles_increase = final_cycles - initial_cycles
    
    print(f"   Initial evolution cycles: {initial_cycles}")
    print(f"   Final evolution cycles: {final_cycles}")
    print(f"   Cycles increase: {cycles_increase}")
    
    # Overall assessment
    print(f"\n📋 Overall Assessment:")
    
    success_criteria = {
        "Non-zero actions": non_zero_actions >= 8,  # Most actions should be non-zero
        "Field evolution": cycles_increase > 0,     # Evolution should continue
        "Prediction tracking": len(prediction_history) > 0  # Should track predictions
    }
    
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion}: {status}")
    
    all_passed = all(success_criteria.values())
    if all_passed:
        print(f"\n🎉 SUCCESS: Prediction-driven learning mechanics working!")
        return True
    else:
        print(f"\n⚠️ PARTIAL: Some prediction-driven learning issues remain")
        return False

if __name__ == "__main__":
    success = test_prediction_driven_learning()
    exit(0 if success else 1)