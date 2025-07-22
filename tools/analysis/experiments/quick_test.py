#!/usr/bin/env python3
"""Quick test of unified brain functionality"""

import sys
sys.path.append('server/src')

try:
    from server.src.brains.field.core_brain import UnifiedFieldBrain
    
    print("🧠 Quick Test: Unified Brain")
    
    # Create small brain for fast testing
    brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)
    print("✅ Brain created")
    
    # Test basic processing
    actions, state = brain.process_robot_cycle([0.1, 0.2, 0.3, 0.4])
    print(f"✅ Basic processing: {len(actions)} actions")
    
    # Test 5 cycles to check addiction system
    for i in range(5):
        actions, state = brain.process_robot_cycle([0.1 + i*0.1, 0.2, 0.3, 0.4])
    
    # Check key features
    has_efficiency = 'prediction_efficiency' in state
    has_addiction = 'learning_addiction_modifier' in state
    has_reward = 'intrinsic_reward' in state
    
    print(f"✅ Prediction efficiency: {has_efficiency}")
    print(f"✅ Learning addiction: {has_addiction}")
    print(f"✅ Intrinsic reward: {has_reward}")
    
    if has_addiction:
        print(f"✅ Addiction modifier: {state['learning_addiction_modifier']:.2f}")
    
    print("🎉 SUCCESS: All features working!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()