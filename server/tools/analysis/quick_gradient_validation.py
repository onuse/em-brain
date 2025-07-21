#!/usr/bin/env python3
"""
Quick Gradient Validation

Quick test to validate the gradient fix works.
"""

import sys
import os
sys.path.append('/Users/jkarlsson/Documents/Projects/robot-project/brain/server/src')

import torch
import math

from brains.field.core_brain import UnifiedFieldBrain

def quick_validation():
    """Quick validation test."""
    
    print("⚡ Quick Gradient Fix Validation")
    print("=" * 35)
    
    # Small brain for fast testing
    brain = UnifiedFieldBrain(
        spatial_resolution=8,  # Small for speed
        temporal_window=5.0,
        quiet_mode=True
    )
    
    print(f"Parameters: decay={brain.field_decay_rate}, "
          f"strength={brain.gradient_following_strength}")
    
    # Test 5 cycles
    non_zero_count = 0
    action_magnitudes = []
    
    for cycle in range(5):
        # Varied sensory input
        sensory_input = [0.5 + 0.1 * cycle * (i % 3) for i in range(24)]
        
        # Process
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Check magnitude
        magnitude = sum(abs(x) for x in motor_output)
        action_magnitudes.append(magnitude)
        
        if magnitude > 1e-8:
            non_zero_count += 1
        
        print(f"Cycle {cycle+1}: magnitude={magnitude:.8f}")
    
    avg_magnitude = sum(action_magnitudes) / len(action_magnitudes)
    
    print(f"\nResults:")
    print(f"Non-zero actions: {non_zero_count}/5 ({non_zero_count*20}%)")
    print(f"Average magnitude: {avg_magnitude:.8f}")
    
    if non_zero_count >= 4 and avg_magnitude > 1e-6:
        print(f"✅ GRADIENT FIX WORKING!")
        return True
    else:
        print(f"❌ Still has issues")
        return False

if __name__ == "__main__":
    try:
        success = quick_validation()
        
        if success:
            print(f"\n🎉 SUCCESS: Brain generates non-zero motor actions!")
            print(f"The dimensional averaging fix resolved the issue.")
        else:
            print(f"\n⚠️  May need further investigation.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()