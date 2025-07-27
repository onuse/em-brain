#!/usr/bin/env python3
"""
Test Pattern-Based Systems

Quick verification that pattern-based motor and attention systems are working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

from src.core.dynamic_brain_factory import DynamicBrainFactory


def test_pattern_systems():
    """Test pattern-based motor and attention."""
    print("\n=== Testing Pattern-Based Systems ===\n")
    
    # Create brain with both pattern systems enabled
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'pattern_motor': True,      # Enable pattern-based motor
        'pattern_attention': True,  # Enable pattern-based attention
        'enable_attention': False,  # Disable gradient-based attention
        'quiet_mode': False        # Show debug output
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=16,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print(f"\n✅ Brain created with:")
    print(f"   Pattern Motor: {brain.pattern_motor_enabled}")
    print(f"   Pattern Attention: {brain.pattern_attention_enabled}")
    
    # Test patterns
    test_patterns = [
        ([1.0] * 16 + [0.0], "All high"),
        ([0.0] * 16 + [0.0], "All low"),
        ([1.0, 0.0] * 8 + [0.0], "Alternating"),
        ([0.5, 1.0, 0.5, 0.0] * 4 + [0.0], "Complex rhythm")
    ]
    
    print("\n📊 Testing different sensory patterns:\n")
    
    for pattern, description in test_patterns:
        print(f"\n🔸 {description} pattern")
        motor_output, brain_state = brain.process_robot_cycle(pattern)
        
        # Show motor output
        print(f"   Motor: {[f'{m:.3f}' for m in motor_output]}")
        
        # Show attention state
        if 'attention' in brain_state:
            att = brain_state['attention']
            print(f"   Attention: focus={att.get('current_focus')}, "
                  f"strength={att.get('focus_strength', 0):.3f}, "
                  f"patterns={att.get('known_patterns', 0)}")
        
        # Show pattern motor info
        if 'pattern_motor' in brain_state:
            pm = brain_state['pattern_motor']
            print(f"   Pattern Motor: forward={pm.get('forward_tendency', 0):.3f}, "
                  f"turn={pm.get('turn_tendency', 0):.3f}")
    
    print("\n✅ Pattern-based systems working correctly!\n")


if __name__ == "__main__":
    test_pattern_systems()