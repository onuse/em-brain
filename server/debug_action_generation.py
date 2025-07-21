#!/usr/bin/env python3
"""
Debug Action Generation Pipeline

Test exactly why the brain produces 0.000 actions despite field evolution.
Trace the field→gradient→action pathway step by step.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_factory import BrainFactory

def debug_action_pipeline():
    """Debug the field→gradient→action generation pipeline."""
    print("🔍 Debugging Action Generation Pipeline...")
    
    # Clear memory
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    
    # Test with diverse input
    pattern_A = [0.8, 0.2, 0.9, 0.1] * 4  # Strong alternating pattern
    
    print(f"\\n🧠 Processing input: {pattern_A[:4]}...")
    action, brain_state = brain.process_sensory_input(pattern_A)
    
    print(f"\\n📊 Brain State Analysis:")
    print(f"   Field energy: {brain_state.get('field_energy', 0.0):.6f}")
    print(f"   Prediction confidence: {brain_state.get('prediction_confidence', 0.0):.6f}")
    print(f"   Processing mode: {brain_state.get('processing_mode', 'unknown')}")
    
    print(f"\\n🎯 Action Output Analysis:")
    print(f"   Raw action: {[f'{x:.9f}' for x in action]}")
    print(f"   Action range: {min(action):.9f} to {max(action):.9f}")
    print(f"   Action magnitude: {np.linalg.norm(action):.9f}")
    print(f"   All zeros? {all(abs(x) < 1e-10 for x in action)}")
    
    # Try to access field implementation directly
    try:
        if hasattr(brain, 'field_brain_adapter'):
            field_brain = brain.field_brain_adapter.field_brain
            print(f"\\n🌊 Field Brain Analysis:")
            print(f"   Field implementation: {type(field_brain.field_impl).__name__}")
            
            # Check if gradients exist
            if hasattr(field_brain.field_impl, 'gradient_flows'):
                gradients = field_brain.field_impl.gradient_flows
                print(f"   Gradient flows: {list(gradients.keys())}")
                
                for key, grad_tensor in gradients.items():
                    grad_magnitude = torch.norm(grad_tensor).item()
                    grad_max = torch.max(torch.abs(grad_tensor)).item()
                    grad_mean = torch.mean(grad_tensor).item()
                    print(f"   {key}: magnitude={grad_magnitude:.9f}, max={grad_max:.9f}, mean={grad_mean:.9f}")
            
            # Test field output generation directly
            print(f"\\n🔧 Direct Field Output Test:")
            center_coords = torch.zeros(field_brain.total_dimensions)
            try:
                direct_output = field_brain.field_impl.generate_field_output(center_coords)
                print(f"   Direct output: {[f'{x:.9f}' for x in direct_output.tolist()]}")
                print(f"   Direct magnitude: {torch.norm(direct_output).item():.9f}")
            except Exception as e:
                print(f"   Direct output failed: {e}")
            
            # Check field state
            field_stats = field_brain.field_impl.get_field_statistics()
            print(f"\\n📈 Field Statistics:")
            for key, value in field_stats.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"❌ Field access failed: {e}")
    
    brain.finalize_session()

def test_multiple_inputs():
    """Test with multiple different inputs to see if any produce non-zero actions."""
    print("\\n" + "="*60)
    print("🧪 Testing Multiple Input Patterns")
    print("="*60)
    
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    brain = BrainFactory(quiet_mode=True)
    
    test_patterns = [
        ([1.0, 0.0] * 8, "High contrast"),
        ([0.1, 0.9] * 8, "Inverted contrast"), 
        ([0.5, 0.5] * 8, "Uniform medium"),
        ([0.0, 0.0] * 8, "All zeros"),
        ([1.0, 1.0] * 8, "All ones"),
        (list(np.random.rand(16)), "Random pattern")
    ]
    
    for pattern, description in test_patterns:
        action, _ = brain.process_sensory_input(pattern)
        magnitude = np.linalg.norm(action)
        print(f"   {description:15}: magnitude={magnitude:.9f}, actions={[f'{x:.6f}' for x in action[:2]]}")
    
    brain.finalize_session()

if __name__ == "__main__":
    debug_action_pipeline()
    test_multiple_inputs()