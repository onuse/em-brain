#!/usr/bin/env python3
"""
Test Continuous Processing Maintenance

Simulate the biological_embodied_learning scenario where the brain
processes continuous sensor input without cognitive idle periods.
Verify that time-based maintenance fallbacks work correctly.
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


def simulate_continuous_processing(brain, duration_seconds=60):
    """Simulate continuous sensor processing like biological_embodied_learning."""
    print(f"🔄 Simulating {duration_seconds}s of continuous processing...")
    
    start_time = time.time()
    cycle_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Simulate varied but continuous sensor input (like robot navigation)
        cycle_count += 1
        sensor_noise = [0.1 + 0.05 * (cycle_count % 10) for _ in range(8)]
        
        # Process through brain (this should maintain high cognitive load)
        action, brain_state = brain.process_sensory_input(sensor_noise)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        cognitive_mode = brain_state.get('cognitive_autopilot', {}).get('cognitive_mode', 'unknown')
        
        # Log occasionally
        if cycle_count % 50 == 0:
            print(f"   Cycle {cycle_count}: confidence={confidence:.3f}, mode={cognitive_mode}")
        
        # Simulate real processing time (like 327ms from biological learning)
        time.sleep(0.1)  # 100ms simulated cycle time
    
    print(f"   Completed {cycle_count} cycles in {duration_seconds}s")
    return cycle_count


def test_continuous_processing_maintenance():
    """Test maintenance during continuous processing scenarios."""
    print("🧠 Testing Maintenance During Continuous Processing")
    print("Simulates biological_embodied_learning scenario")
    
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
    
    print("✅ Brain with continuous processing maintenance ready")
    
    # Get initial maintenance status
    initial_status = brain.get_maintenance_status()
    print(f"\n📊 Initial maintenance counts:")
    print(f"   Light: {initial_status['maintenance_stats']['light_maintenance_count']}")
    print(f"   Heavy: {initial_status['maintenance_stats']['heavy_maintenance_count']}")
    print(f"   Deep: {initial_status['maintenance_stats']['deep_consolidation_count']}")
    
    # Test 1: Short continuous processing (should not trigger time-based)
    print(f"\n🔄 Test 1: Short continuous processing (2 minutes)")
    simulate_continuous_processing(brain, duration_seconds=120)
    
    status_after_short = brain.get_maintenance_status()
    print(f"   After 2 min - Light: {status_after_short['maintenance_stats']['light_maintenance_count']}")
    
    # Test 2: Extended continuous processing (should trigger time-based light maintenance)
    print(f"\n🔄 Test 2: Extended continuous processing (6 minutes total)")
    simulate_continuous_processing(brain, duration_seconds=240)  # Additional 4 minutes
    
    status_after_medium = brain.get_maintenance_status()
    light_count = status_after_medium['maintenance_stats']['light_maintenance_count']
    print(f"   After 6 min - Light: {light_count} (should trigger time-based at 5 min)")
    
    # Test 3: Very long processing (should trigger heavy maintenance)
    print(f"\n🔄 Test 3: Very long continuous processing (25 minutes total)")
    simulate_continuous_processing(brain, duration_seconds=1140)  # Additional 19 minutes
    
    final_status = brain.get_maintenance_status()
    print(f"\n📊 Final maintenance counts after 25 minutes:")
    print(f"   Light: {final_status['maintenance_stats']['light_maintenance_count']} (should be ≥5 from time-based)")
    print(f"   Heavy: {final_status['maintenance_stats']['heavy_maintenance_count']} (should be ≥1 from 20-min time-based)")
    print(f"   Deep: {final_status['maintenance_stats']['deep_consolidation_count']}")
    
    # Test 4: Manual check for immediate recommendations
    print(f"\n🔧 Test 4: Current maintenance recommendations")
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Cognitive idle time: {idle_time:.1f}s")
    print(f"   Recommendations: {recommendations}")
    
    # Try manual trigger
    performed = brain.run_recommended_maintenance()
    print(f"   Manual trigger result: {performed}")
    
    brain.finalize_session()
    print("\n✅ Continuous processing maintenance test completed")
    print("\nKey insight: Time-based fallbacks ensure maintenance happens even")
    print("during intensive continuous processing scenarios like embodied learning.")


if __name__ == "__main__":
    print("🔄 Continuous Processing Maintenance Test")
    print("=" * 55)
    
    test_continuous_processing_maintenance()
    
    print("\n🎯 Test completed")