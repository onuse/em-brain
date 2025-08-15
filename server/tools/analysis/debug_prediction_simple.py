#!/usr/bin/env python3
"""
Simple debug of prediction learning - check if confidence ever changes
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add paths
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))
testing_path = Path(__file__).parent.parent / 'testing'
sys.path.insert(0, str(testing_path))

from behavioral_test_dynamic import DynamicBehavioralTestFramework

def simple_prediction_debug():
    """Simple check of prediction confidence"""
    print("🔍 Simple Prediction Debug")
    print("=" * 60)
    
    framework = DynamicBehavioralTestFramework(quiet_mode=True)
    framework.setup_virtual_robot()
    
    print("\nSending repeating pattern to brain...")
    
    # Simple repeating pattern
    pattern = [0.5, 0.8, 0.3, 0.6] * 4  # 16D
    
    for i in range(50):
        motor_output = framework.connection_handler.handle_sensory_input(
            framework.client_id, pattern
        )
        
        # Every 10 cycles, check what's happening
        if i % 10 == 0:
            print(f"\nCycle {i}:")
            print(f"  Motor output: [{motor_output[0]:.3f}, {motor_output[1]:.3f}, ...]")
            
            # Try to get brain state through connection handler stats
            stats = framework.connection_handler.get_stats()
            print(f"  Total messages: {stats['total_messages']}")
            
            # Check if we have active sessions
            if stats.get('active_sessions'):
                for session_info in stats['active_sessions']:
                    print(f"  Session {session_info['session_id']}:")
                    print(f"    Cycles: {session_info.get('cycles_processed', 0)}")
                    print(f"    Avg time: {session_info.get('average_cycle_time_ms', 0):.1f}ms")
    
    # Now let's test with a different pattern
    print("\n" + "=" * 60)
    print("Testing with different pattern...")
    
    pattern2 = [0.2, 0.9, 0.7, 0.1] * 4
    
    motor_outputs = []
    for i in range(20):
        motor_output = framework.connection_handler.handle_sensory_input(
            framework.client_id, pattern2
        )
        motor_outputs.append(motor_output[:4])
    
    # Check if outputs are changing
    motor_outputs = np.array(motor_outputs)
    variance = np.var(motor_outputs, axis=0)
    print(f"\nMotor output variance: {variance}")
    print(f"Average variance: {np.mean(variance):.6f}")
    
    if np.mean(variance) < 0.0001:
        print("❌ Motor outputs are not changing - brain may be stuck")
    else:
        print("✅ Motor outputs are changing")
    
    framework.cleanup()

if __name__ == "__main__":
    simple_prediction_debug()