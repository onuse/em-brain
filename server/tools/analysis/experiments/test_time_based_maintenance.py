#!/usr/bin/env python3
"""
Test Time-Based Maintenance Triggers

Quick test to verify that time-based maintenance fallbacks work
for continuous processing scenarios.
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


def test_time_based_maintenance():
    """Test time-based maintenance fallback triggers."""
    print("🧠 Testing Time-Based Maintenance Fallbacks")
    
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
    
    print("✅ Brain with time-based maintenance ready")
    
    # Reset maintenance timestamps to simulate long-running scenario
    print("\n🕒 Simulating brain that has been running for 6 minutes without maintenance...")
    brain.maintenance_scheduler.brain.last_light_maintenance = time.time() - 360  # 6 minutes ago
    brain.maintenance_scheduler.brain.last_heavy_maintenance = time.time() - 1500  # 25 minutes ago  
    brain.maintenance_scheduler.brain.last_deep_consolidation = time.time() - 4000  # 67 minutes ago
    
    # Process some sensor input to maintain high cognitive load
    print("🔄 Processing sensor input (high cognitive load)...")
    for i in range(3):
        sensor_input = [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4]  # Novel patterns
        action, brain_state = brain.process_sensory_input(sensor_input)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        cognitive_mode = brain_state.get('cognitive_autopilot', {}).get('cognitive_mode', 'unknown')
        print(f"   Cycle {i+1}: confidence={confidence:.3f}, mode={cognitive_mode}")
        time.sleep(0.5)
    
    # Check recommendations
    print(f"\n🔧 Checking maintenance recommendations...")
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    
    print(f"   Cognitive idle time: {idle_time:.1f}s")
    print(f"   Time since last light: {(time.time() - brain.maintenance_scheduler.brain.last_light_maintenance)/60:.1f} min")
    print(f"   Time since last heavy: {(time.time() - brain.maintenance_scheduler.brain.last_heavy_maintenance)/60:.1f} min")  
    print(f"   Time since last deep: {(time.time() - brain.maintenance_scheduler.brain.last_deep_consolidation)/60:.1f} min")
    print(f"   Recommendations: {recommendations}")
    
    # Try to trigger maintenance
    print(f"\n⚡ Attempting to trigger maintenance...")
    performed = brain.run_recommended_maintenance()
    print(f"   Maintenance performed: {performed}")
    
    if any(performed.values()):
        print("✅ Time-based maintenance fallbacks working!")
    else:
        print("❌ Time-based maintenance fallbacks not working")
    
    # Check final status
    status = brain.get_maintenance_status()
    print(f"\n📊 Final maintenance counts:")
    print(f"   Light: {status['maintenance_stats']['light_maintenance_count']}")
    print(f"   Heavy: {status['maintenance_stats']['heavy_maintenance_count']}")
    print(f"   Deep: {status['maintenance_stats']['deep_consolidation_count']}")
    
    brain.finalize_session()
    print("\n✅ Time-based maintenance test completed")


if __name__ == "__main__":
    print("🕒 Time-Based Maintenance Test")
    print("=" * 40)
    
    test_time_based_maintenance()
    
    print("\n🎯 Test completed")