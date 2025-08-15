#!/usr/bin/env python3
"""
Debug prediction learning in the dynamic brain

Investigates why prediction confidence remains at 0.0
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

# Add testing path for behavioral test framework
testing_path = Path(__file__).parent.parent / 'testing'
sys.path.insert(0, str(testing_path))

from behavioral_test_dynamic import DynamicBehavioralTestFramework

def debug_prediction_learning():
    """Deep dive into prediction learning behavior"""
    print("🔍 Debugging Prediction Learning")
    print("=" * 60)
    
    # Create framework with verbose output
    framework = DynamicBehavioralTestFramework(
        use_simple_brain=False,
        quiet_mode=False  # Want to see what's happening
    )
    
    # Setup virtual robot
    framework.setup_virtual_robot()
    
    print("\n📊 Running detailed prediction analysis...")
    print("-" * 60)
    
    # Create predictable sine wave pattern
    for cycle in range(20):
        phase = cycle * 0.1
        sensory_input = [
            np.sin(phase),
            np.cos(phase),
            np.sin(phase * 2),
            np.cos(phase * 2)
        ] + [0.1] * 12  # Pad to 16D
        
        print(f"\nCycle {cycle}:")
        print(f"  Input pattern: [{sensory_input[0]:.3f}, {sensory_input[1]:.3f}, ...]")
        
        # Get motor output and session info
        motor_output = framework.connection_handler.handle_sensory_input(
            framework.client_id, sensory_input
        )
        
        # Get session to check brain state
        sessions = framework.brain_service.list_sessions()
        if sessions:
            session = framework.brain_service.get_session(sessions[0].session_id)
            if session and hasattr(session, 'brain'):
                brain = session.brain
                
                # Check internal state
                print(f"  Brain cycles: {brain.brain_cycles}")
                print(f"  Field energy: {brain.get_field_statistics()['field_energy']:.6f}")
                
                # Check prediction confidence
                if hasattr(brain, '_current_prediction_confidence'):
                    print(f"  Prediction confidence: {brain._current_prediction_confidence:.3f}")
                else:
                    print(f"  WARNING: No prediction confidence attribute")
                
                # Check if brain has memory regions
                if hasattr(brain, 'memory_regions'):
                    print(f"  Memory regions: {len(brain.memory_regions)}")
                
                # Check if experiences are being stored
                if hasattr(brain, 'experiences'):
                    print(f"  Stored experiences: {len(brain.experiences)}")
                
                # Check blended reality system
                if hasattr(brain, 'blended_reality'):
                    blend_state = brain.blended_reality.get_blend_state()
                    print(f"  Reality blend: {blend_state['reality_weight']:.1%} reality / {blend_state['fantasy_weight']:.1%} fantasy")
                    print(f"  Dream state: {blend_state['is_dreaming']}")
        
        print(f"  Motor output: [{motor_output[0]:.3f}, {motor_output[1]:.3f}, ...]")
        
        # Small delay to make output readable
        time.sleep(0.1)
    
    # Final analysis
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Check what components are active
    if sessions:
        session = framework.brain_service.get_session(sessions[0].session_id)
        if session and hasattr(session, 'brain'):
            brain = session.brain
            
            print("\nBrain Configuration:")
            print(f"  Type: {type(brain).__name__}")
            print(f"  Field dimensions: {brain.get_field_dimensions()}D")
            
            # Check which systems are enabled
            if hasattr(brain, 'use_integrated_attention'):
                print(f"  Integrated attention: {'ENABLED' if brain.use_integrated_attention else 'DISABLED'}")
            if hasattr(brain, 'use_blended_reality'):
                print(f"  Blended reality: {'ENABLED' if brain.use_blended_reality else 'DISABLED'}")
            if hasattr(brain, 'use_enhanced_dynamics'):
                print(f"  Enhanced dynamics: {'ENABLED' if brain.use_enhanced_dynamics else 'DISABLED'}")
            
            # Check prediction-related components
            print("\nPrediction Components:")
            if hasattr(brain, 'prediction_system'):
                print(f"  Prediction system: PRESENT")
            else:
                print(f"  Prediction system: MISSING")
            
            if hasattr(brain, '_calculate_prediction_confidence'):
                print(f"  Prediction calculation method: PRESENT")
            else:
                print(f"  Prediction calculation method: MISSING")
    
    framework.cleanup()

if __name__ == "__main__":
    debug_prediction_learning()