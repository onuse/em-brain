#!/usr/bin/env python3
"""
CPU-Optimized Behavioral Test for Unified Brain
Very small spatial resolution for fast CPU testing.
"""

import sys
import os
sys.path.append('server/src')

from server.src.brain_factory import BrainFactory
import numpy as np

def test_cpu_optimized_learning():
    """Test basic learning with very small field for CPU speed."""
    print("🚀 CPU-Optimized Behavioral Test")
    print("   Using minimal spatial resolution for speed")
    
    # CPU-optimized configuration: very small field
    config = {
        'brain': {
            'type': 'field', 
            'sensory_dim': 4,  # Reduced from 16
            'motor_dim': 4,
            'spatial_resolution': 5  # Reduced from 20 - only 25 field elements
        },
        'memory': {'enable_persistence': False}
    }
    
    brain = BrainFactory(config=config, quiet_mode=True)
    
    # Simple repeating pattern
    pattern = [0.1, 0.3, 0.7, 0.9]
    
    print(f"Testing {len(pattern)}D pattern with 5×5 field (25 elements total)")
    
    confidences = []
    start_time = time.time()
    
    # Only 20 cycles for speed
    for i in range(20):
        action, brain_state = brain.process_sensory_input(pattern)
        
        confidence = brain_state.get('last_action_confidence', 
                                   brain_state.get('prediction_confidence', 0.0))
        confidences.append(confidence)
        
        if i < 3:  # Show first few cycles
            print(f"  Cycle {i+1}: confidence={confidence:.3f}, actions={[f'{x:.2f}' for x in action]}")
    
    elapsed = time.time() - start_time
    cycles_per_second = 20 / elapsed
    
    # Analysis
    early_conf = np.mean(confidences[:5])
    late_conf = np.mean(confidences[-5:])
    improvement = late_conf - early_conf
    
    print(f"\n📊 Results:")
    print(f"  Performance: {cycles_per_second:.1f} cycles/sec ({elapsed:.1f}s total)")
    print(f"  Early confidence: {early_conf:.3f}")
    print(f"  Late confidence: {late_conf:.3f}")
    print(f"  Learning improvement: {improvement:.3f}")
    
    success = improvement > 0.05 or late_conf > 0.2
    print(f"  Learning detected: {'✅' if success else '❌'}")
    
    if cycles_per_second > 1.0:
        print(f"  CPU performance: {'✅ Acceptable' if cycles_per_second > 2.0 else '⚠️ Slow but workable'}")
    else:
        print(f"  CPU performance: ❌ Too slow ({cycles_per_second:.1f} cps)")
    
    return success

if __name__ == "__main__":
    import time
    test_cpu_optimized_learning()