#!/usr/bin/env python3
"""
Test Autopilot-Based Maintenance

Demonstrate maintenance scheduling based on cognitive autopilot modes:
- AUTOPILOT mode (high confidence): Brain coasting, maintenance encouraged
- FOCUSED mode (medium confidence): Moderate thinking, light maintenance OK  
- DEEP_THINK mode (low confidence): Intensive thinking, avoid maintenance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory


def simulate_sensor_scenario(brain, description, input_pattern, cycles=3):
    """Simulate a sensor input scenario and show cognitive mode response."""
    print(f"\n🔄 {description}")
    
    cognitive_modes = []
    confidences = []
    
    for i in range(cycles):
        action, brain_state = brain.process_sensory_input(input_pattern)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        cognitive_mode = brain_state.get('cognitive_autopilot', {}).get('cognitive_mode', 'unknown')
        mode_changed = brain_state.get('cognitive_autopilot', {}).get('mode_changed', False)
        
        cognitive_modes.append(cognitive_mode)
        confidences.append(confidence)
        
        mode_indicator = " ⭐" if mode_changed else ""
        print(f"   Cycle {i+1}: confidence={confidence:.3f}, mode={cognitive_mode}{mode_indicator}")
        
        time.sleep(0.5)
    
    return cognitive_modes, confidences


def test_autopilot_maintenance():
    """Test maintenance based on cognitive autopilot state."""
    print("🧠 Testing Autopilot-Based Maintenance Scheduling")
    print("Shows how maintenance triggers during different cognitive modes")
    
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
    
    print("✅ Brain with cognitive autopilot-based maintenance ready")
    
    # Scenario 1: Novel input (should trigger DEEP_THINK mode)
    novel_modes, novel_conf = simulate_sensor_scenario(
        brain, 
        "Novel/unpredictable input (expect DEEP_THINK mode)",
        [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],  # Varied input
        cycles=5
    )
    
    print(f"   Average confidence: {sum(novel_conf)/len(novel_conf):.3f}")
    print(f"   Dominant mode: {max(set(novel_modes), key=novel_modes.count)}")
    
    # Check maintenance during deep thinking
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Maintenance during deep thinking - recommendations: {recommendations}")
    
    # Scenario 2: Repetitive input (should build confidence, move toward AUTOPILOT)
    print("\n⏳ Waiting 10 seconds, then starting repetitive pattern...")
    time.sleep(10.0)  # Allow some idle time accumulation
    
    repetitive_modes, repetitive_conf = simulate_sensor_scenario(
        brain,
        "Repetitive/predictable input (expect confidence increase)",
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Same input repeated
        cycles=8
    )
    
    print(f"   Average confidence: {sum(repetitive_conf)/len(repetitive_conf):.3f}")
    print(f"   Dominant mode: {max(set(repetitive_modes), key=repetitive_modes.count)}")
    
    # Check maintenance during autopilot mode
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Maintenance during autopilot - recommendations: {recommendations}")
    
    # Try to trigger maintenance
    print(f"   Attempting maintenance during autopilot mode...")
    performed = brain.run_recommended_maintenance()
    print(f"   Maintenance performed: {performed}")
    
    # Scenario 3: Return to novel input (test mode switching)
    novel2_modes, novel2_conf = simulate_sensor_scenario(
        brain,
        "Return to novel input (test mode switching)",
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],  # Different varied pattern
        cycles=4
    )
    
    print(f"   Average confidence: {sum(novel2_conf)/len(novel2_conf):.3f}")
    print(f"   Dominant mode: {max(set(novel2_modes), key=novel2_modes.count)}")
    
    # Final maintenance status
    print(f"\n📊 Final Results:")
    
    # Get cognitive autopilot stats
    if hasattr(brain.cognitive_autopilot, 'time_in_modes'):
        print(f"   Time in modes:")
        for mode, duration in brain.cognitive_autopilot.time_in_modes.items():
            print(f"     {mode.value}: {duration:.1f}s")
    
    # Get maintenance stats
    status = brain.get_maintenance_status()
    print(f"   Maintenance performed:")
    print(f"     Light: {status['maintenance_stats']['light_maintenance_count']}")
    print(f"     Heavy: {status['maintenance_stats']['heavy_maintenance_count']}")
    print(f"     Deep: {status['maintenance_stats']['deep_consolidation_count']}")
    
    brain.finalize_session()
    print("\n✅ Autopilot-based maintenance test completed")
    print("\nKey insight: Maintenance now intelligently schedules based on cognitive modes.")
    print("Brain can process continuous sensor input while performing maintenance")
    print("during autopilot (coasting) periods, avoiding disruption during deep thinking.")


if __name__ == "__main__":
    print("🧠 Cognitive Autopilot Maintenance Test")
    print("=" * 65)
    
    test_autopilot_maintenance()
    
    print("\n🎯 Test completed")