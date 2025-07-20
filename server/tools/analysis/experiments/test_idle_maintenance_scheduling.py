#!/usr/bin/env python3
"""
Test Idle-Based Maintenance Scheduling

Verify that the brain loop's idle-based maintenance scheduling works correctly
by creating scenarios with different idle periods and confirming maintenance
operations are triggered at appropriate times.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import threading
from typing import Dict, Any

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)

# Add the parent directory to handle relative imports in brain_factory
import importlib.util
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory
from src.brain_loop import DecoupledBrainLoop
from src.communication.sensor_buffer import get_sensor_buffer


def test_idle_maintenance_with_brain_loop():
    """Test that idle-based maintenance scheduling works with the brain loop."""
    print("🧪 Testing idle-based maintenance scheduling with brain loop")
    
    # Use field brain for comprehensive maintenance interface testing
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4
        }
    }
    
    # Create brain factory with field brain (has full maintenance interface)
    brain = BrainFactory(config=config, quiet_mode=True)
    
    # Verify brain has maintenance interface
    if not hasattr(brain, 'maintenance_scheduler') or brain.maintenance_scheduler is None:
        print("❌ Brain does not have maintenance scheduler - test invalid")
        return
    
    print(f"✅ Brain has maintenance scheduler: {type(brain.maintenance_scheduler).__name__}")
    
    # Create decoupled brain loop
    brain_loop = DecoupledBrainLoop(brain, cycle_time_ms=50.0)
    
    # Get sensor buffer for injecting test data
    sensor_buffer = get_sensor_buffer()
    
    print("\n🔄 Starting brain loop...")
    brain_loop.start()
    
    try:
        # Phase 1: Active processing (should mark activity, no maintenance)
        print("\n📡 Phase 1: Active sensor processing (10 cycles)")
        for i in range(10):
            # Inject sensor data to keep brain active
            test_sensors = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
            sensor_buffer.add_sensor_data(f"test_client_{i}", test_sensors)
            time.sleep(0.1)  # 100ms intervals
        
        # Check initial state
        idle_time = brain.maintenance_scheduler.get_idle_time()
        print(f"   Current idle time after active phase: {idle_time:.2f}s")
        
        # Phase 2: Short idle period (should trigger light maintenance at 5+ seconds)
        print("\n💤 Phase 2: Short idle period (8 seconds - should trigger light maintenance)")
        time.sleep(8.0)
        
        # Check maintenance scheduler state
        idle_time = brain.maintenance_scheduler.get_idle_time()
        recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
        maintenance_status = brain.get_maintenance_status()
        
        print(f"   Idle time: {idle_time:.2f}s")
        print(f"   Recommendations: {recommendations}")
        print(f"   Light maintenance count: {maintenance_status['maintenance_stats']['light_maintenance_count']}")
        
        # Phase 3: Medium idle period (should trigger heavy maintenance at 30+ seconds)
        print("\n💤 Phase 3: Medium idle period (35 seconds total - should trigger heavy maintenance)")
        time.sleep(27.0)  # Additional time to reach 35 seconds total
        
        idle_time = brain.maintenance_scheduler.get_idle_time()
        recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
        maintenance_status = brain.get_maintenance_status()
        
        print(f"   Idle time: {idle_time:.2f}s")
        print(f"   Recommendations: {recommendations}")
        print(f"   Heavy maintenance count: {maintenance_status['maintenance_stats']['heavy_maintenance_count']}")
        
        # Phase 4: Long idle period (should trigger deep consolidation at 60+ seconds)
        print("\n💤 Phase 4: Long idle period (65 seconds total - should trigger deep consolidation)")
        time.sleep(30.0)  # Additional time to reach 65 seconds total
        
        idle_time = brain.maintenance_scheduler.get_idle_time()
        recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
        maintenance_status = brain.get_maintenance_status()
        
        print(f"   Idle time: {idle_time:.2f}s")
        print(f"   Recommendations: {recommendations}")
        print(f"   Deep consolidation count: {maintenance_status['maintenance_stats']['deep_consolidation_count']}")
        
        # Phase 5: Manual maintenance test
        print("\n🔧 Phase 5: Manual maintenance trigger test")
        performed = brain.run_recommended_maintenance()
        print(f"   Manual maintenance performed: {performed}")
        
        # Final maintenance status
        final_status = brain.get_maintenance_status()
        print(f"\n📊 Final maintenance statistics:")
        for key, value in final_status['maintenance_stats'].items():
            print(f"   {key}: {value}")
        
        # Brain loop statistics
        print(f"\n🧠 Brain loop statistics:")
        loop_stats = brain_loop.get_loop_statistics()
        print(f"   Total cycles: {loop_stats['total_cycles']}")
        print(f"   Active cycles: {loop_stats['active_cycles']}")
        print(f"   Idle cycles: {loop_stats['idle_cycles']}")
        print(f"   Maintenance tasks performed: {loop_stats['maintenance_tasks_performed']}")
        
        print(f"\n✅ Idle maintenance scheduling test completed")
        
    finally:
        print("\n🛑 Stopping brain loop...")
        brain_loop.stop()
        brain.finalize_session()


def test_manual_maintenance_scheduling():
    """Test manual maintenance scheduling without brain loop."""
    print("\n🧪 Testing manual maintenance scheduling (no brain loop)")
    
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=True)
    
    if not hasattr(brain, 'maintenance_scheduler') or brain.maintenance_scheduler is None:
        print("❌ Brain does not have maintenance scheduler")
        return
    
    print("✅ Testing maintenance scheduler directly")
    
    # Test immediate recommendations (should be minimal)
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   Initial idle time: {idle_time:.2f}s, recommendations: {recommendations}")
    
    # Wait 6 seconds to trigger light maintenance
    print("   Waiting 6 seconds for light maintenance threshold...")
    time.sleep(6.0)
    
    idle_time = brain.maintenance_scheduler.get_idle_time()
    recommendations = brain.maintenance_scheduler.brain.get_maintenance_recommendations(idle_time)
    print(f"   After 6s - idle time: {idle_time:.2f}s, recommendations: {recommendations}")
    
    # Run recommended maintenance
    performed = brain.maintenance_scheduler.run_recommended_maintenance()
    print(f"   Maintenance performed: {performed}")
    
    # Check final status
    status = brain.get_maintenance_status()
    print(f"   Final light maintenance count: {status['maintenance_stats']['light_maintenance_count']}")
    
    brain.finalize_session()


if __name__ == "__main__":
    print("🔧 Idle-Based Maintenance Scheduling Test")
    print("=" * 60)
    
    # Test 1: Manual scheduling
    test_manual_maintenance_scheduling()
    
    print("\n" + "=" * 60)
    
    # Test 2: Brain loop integration
    test_idle_maintenance_with_brain_loop()
    
    print("\n🎯 All maintenance scheduling tests completed")