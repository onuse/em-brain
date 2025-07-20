#!/usr/bin/env python3
"""
Test Cognitive Load-Based Maintenance

Demonstrate how maintenance scheduling now works based on cognitive load
rather than sensor silence - sensors can always be active, but maintenance
triggers during periods of low cognitive demand (routine processing).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import random

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory


def test_cognitive_load_maintenance():
    """Test maintenance scheduling based on cognitive load, not sensor silence."""
    print("🧪 Testing cognitive load-based maintenance scheduling")
    print("Concept: Sensors always active, maintenance during low cognitive load periods")
    
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=True)
    
    if not brain.maintenance_scheduler:
        print("❌ No maintenance scheduler available")
        return
    
    print("\n🧠 Brain initialized with cognitive load-aware maintenance")
    
    # Scenario 1: High confidence (routine) processing - should allow maintenance
    print("\n📡 Scenario 1: High confidence inputs (routine sensor processing)")
    print("   Simulating familiar/predictable sensor patterns...")
    
    # Send very predictable, boring sensor data
    for i in range(5):
        predictable_input = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Boring, predictable
        action, brain_state = brain.process_sensory_input(predictable_input)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        cognitive_load = 1.0 - confidence
        load_desc = "high" if cognitive_load > 0.7 else "med" if cognitive_load > 0.3 else "low"
        
        print(f"   Cycle {i+1}: confidence={confidence:.3f}, cognitive_load={cognitive_load:.3f} ({load_desc})")
        time.sleep(1.0)
    
    # Check if maintenance would trigger during routine processing
    print("\n   Checking maintenance recommendations after routine processing:")
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Cognitive idle time: {idle_time:.1f}s")
    print(f"   Maintenance recommendations: {recommendations}")
    
    # Try running maintenance
    performed = brain.run_recommended_maintenance()
    print(f"   Maintenance performed: {performed}")
    
    # Scenario 2: Low confidence (novel) processing - should inhibit maintenance
    print("\n🚨 Scenario 2: Low confidence inputs (novel situations)")
    print("   Simulating unpredictable, novel sensor patterns...")
    
    # Reset activity timer 
    brain.maintenance_scheduler.last_activity_time = time.time() - 20.0  # Fake 20s idle
    
    # Send novel, unpredictable sensor data
    for i in range(3):
        novel_input = [random.random() for _ in range(8)]  # Random, unpredictable
        action, brain_state = brain.process_sensory_input(novel_input)
        
        confidence = brain_state.get('prediction_confidence', 0.0) 
        cognitive_load = 1.0 - confidence
        load_desc = "high" if cognitive_load > 0.7 else "med" if cognitive_load > 0.3 else "low"
        
        print(f"   Cycle {i+1}: confidence={confidence:.3f}, cognitive_load={cognitive_load:.3f} ({load_desc})")
        time.sleep(0.5)
    
    # Check maintenance after novel processing
    print("\n   Checking maintenance after novel input processing:")
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Cognitive idle time: {idle_time:.1f}s")
    print(f"   Maintenance recommendations: {recommendations}")
    
    # Scenario 3: Return to routine processing - maintenance should become available again
    print("\n🔄 Scenario 3: Return to routine processing")
    print("   Brain adapts to patterns, cognitive load decreases...")
    
    # Repeat the same pattern to build familiarity
    familiar_pattern = [0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2]
    for i in range(8):
        action, brain_state = brain.process_sensory_input(familiar_pattern)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        cognitive_load = 1.0 - confidence  
        load_desc = "high" if cognitive_load > 0.7 else "med" if cognitive_load > 0.3 else "low"
        
        if i % 2 == 0:  # Log every other cycle
            print(f"   Cycle {i+1}: confidence={confidence:.3f}, cognitive_load={cognitive_load:.3f} ({load_desc})")
        time.sleep(0.5)
    
    # Final maintenance check
    print("\n   Final maintenance check after pattern familiarization:")
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Cognitive idle time: {idle_time:.1f}s")
    print(f"   Maintenance recommendations: {recommendations}")
    
    performed = brain.run_recommended_maintenance()
    print(f"   Maintenance performed: {performed}")
    
    # Summary
    print(f"\n📊 Summary:")
    status = brain.get_maintenance_status()
    print(f"   Total light maintenance: {status['maintenance_stats']['light_maintenance_count']}")
    print(f"   Total heavy maintenance: {status['maintenance_stats']['heavy_maintenance_count']}")
    print(f"   Total deep consolidation: {status['maintenance_stats']['deep_consolidation_count']}")
    
    brain.finalize_session()
    print("\n✅ Cognitive load-based maintenance test completed")
    print("\nKey insight: Maintenance now triggers during routine processing periods,")
    print("not sensor silence. Brain can handle continuous sensor input while still")
    print("performing maintenance during cognitively idle periods.")


if __name__ == "__main__":
    print("🧠 Cognitive Load-Based Maintenance Test")
    print("=" * 60)
    
    test_cognitive_load_maintenance()
    
    print("\n🎯 Test completed")