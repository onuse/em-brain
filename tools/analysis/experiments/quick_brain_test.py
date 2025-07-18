#!/usr/bin/env python3
"""Quick test of basic brain functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

try:
    from src.brain import MinimalBrain
    
    print('🧠 Testing basic brain functionality...')
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Test processing
    for i in range(5):
        sensory = [0.1, 0.2, 0.3, 0.4]  # 4D sensory input
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        print(f'Test {i+1}: {len(sensory)}D → {len(predicted_action)}D action')
    
    print('✅ Basic brain functionality works!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()