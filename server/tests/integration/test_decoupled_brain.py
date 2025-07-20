#!/usr/bin/env python3
"""
Test Decoupled Brain Loop System

Tests the sensor buffer and decoupled brain loop to validate that:
1. Sensor buffer properly stores and retrieves latest sensor data
2. Brain loop runs independently at 50ms intervals
3. System handles idle periods gracefully
4. Performance meets timing requirements
"""

import sys
import os
import time
import threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.brain_factory import MinimalBrain
from server.src.brain_loop import DecoupledBrainLoop
from server.src.communication.sensor_buffer import get_sensor_buffer


def test_sensor_buffer():
    """Test sensor buffer functionality."""
    print("🧪 Testing Sensor Buffer")
    print("-" * 30)
    
    buffer = get_sensor_buffer()
    
    # Test adding sensor data
    client1_data = [1.0, 2.0, 3.0, 4.0]
    client2_data = [5.0, 6.0, 7.0, 8.0]
    
    # Add data for multiple clients
    assert buffer.add_sensor_input("client1", client1_data)
    assert buffer.add_sensor_input("client2", client2_data)
    
    # Check data retrieval
    data1 = buffer.get_latest_sensor_data("client1")
    data2 = buffer.get_latest_sensor_data("client2")
    
    assert data1 is not None
    assert data2 is not None
    assert data1.vector == client1_data
    assert data2.vector == client2_data
    
    # Test data replacement (keep only latest)
    new_client1_data = [10.0, 20.0, 30.0, 40.0]
    buffer.add_sensor_input("client1", new_client1_data)
    
    updated_data1 = buffer.get_latest_sensor_data("client1")
    assert updated_data1.vector == new_client1_data
    
    # Test statistics
    stats = buffer.get_statistics()
    assert stats['total_inputs_received'] == 3
    assert stats['total_inputs_discarded'] == 1  # client1 data was replaced
    assert stats['active_clients'] == 2
    
    print("✅ Sensor buffer tests passed")
    print(f"   Stats: {stats}")


def test_decoupled_brain_loop():
    """Test decoupled brain loop functionality."""
    print("\n🧪 Testing Decoupled Brain Loop")
    print("-" * 35)
    
    # Create brain and loop
    brain = MinimalBrain(quiet_mode=True)
    brain_loop = DecoupledBrainLoop(brain, cycle_time_ms=50.0)
    
    # Test initial state
    assert not brain_loop.is_running()
    assert brain_loop.total_cycles == 0
    
    # Start the loop
    brain_loop.start()
    assert brain_loop.is_running()
    
    # Let it run for a bit with no sensor data (idle cycles)
    print("   Testing idle cycles...")
    time.sleep(0.3)  # 300ms = ~6 cycles
    
    stats = brain_loop.get_loop_statistics()
    print(f"   Idle cycles: {stats['idle_cycles']}")
    assert stats['idle_cycles'] > 3  # Should have some idle cycles
    
    # Add sensor data and test active cycles
    buffer = get_sensor_buffer()
    buffer.add_sensor_input("test_client", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    
    print("   Testing active cycles...")
    time.sleep(0.2)  # 200ms = ~4 cycles
    
    updated_stats = brain_loop.get_loop_statistics()
    print(f"   Active cycles: {updated_stats['active_cycles']}")
    assert updated_stats['active_cycles'] > 0  # Should have processed sensor data
    
    # Test timing performance
    avg_cycle_time = updated_stats['actual_avg_cycle_time_ms']
    target_cycle_time = updated_stats['target_cycle_time_ms']
    
    print(f"   Avg cycle time: {avg_cycle_time:.1f}ms (target: {target_cycle_time:.1f}ms)")
    assert avg_cycle_time < target_cycle_time * 2  # Reasonable performance
    
    # Stop the loop
    brain_loop.stop()
    assert not brain_loop.is_running()
    
    print("✅ Decoupled brain loop tests passed")


def test_concurrent_operation():
    """Test brain loop with concurrent sensor input."""
    print("\n🧪 Testing Concurrent Operation")
    print("-" * 35)
    
    brain = MinimalBrain(quiet_mode=True)
    brain_loop = DecoupledBrainLoop(brain, cycle_time_ms=30.0)  # Faster for testing
    buffer = get_sensor_buffer()
    
    # Start brain loop
    brain_loop.start()
    
    # Simulate multiple clients sending data concurrently
    def simulate_client(client_id: str, duration: float):
        """Simulate a client sending sensor data."""
        end_time = time.time() + duration
        cycle = 0
        
        while time.time() < end_time:
            # Generate varying sensor data
            sensor_data = [
                float(cycle % 10),  # Varying values
                float(client_id[-1]),  # Client-specific
                time.time() % 10.0,  # Time-based
                1.0  # Constant
            ]
            
            buffer.add_sensor_input(client_id, sensor_data)
            cycle += 1
            time.sleep(0.05)  # 50ms between sensor updates
    
    # Start multiple client simulators
    client_threads = []
    for i in range(3):
        thread = threading.Thread(
            target=simulate_client, 
            args=(f"client_{i}", 0.5),  # 500ms duration
            daemon=True
        )
        client_threads.append(thread)
        thread.start()
    
    # Wait for all clients to finish
    for thread in client_threads:
        thread.join()
    
    # Let brain process remaining data
    time.sleep(0.2)
    
    # Check final statistics
    brain_stats = brain_loop.get_loop_statistics()
    buffer_stats = buffer.get_statistics()
    
    print(f"   Brain cycles: {brain_stats['total_cycles']}")
    print(f"   Active cycles: {brain_stats['active_cycles']}")
    print(f"   Buffer inputs: {buffer_stats['total_inputs_received']}")
    print(f"   Buffer efficiency: {buffer_stats['buffer_efficiency']:.1%}")
    
    assert brain_stats['active_cycles'] > 0
    assert buffer_stats['total_inputs_received'] > 0
    
    brain_loop.stop()
    print("✅ Concurrent operation tests passed")


def main():
    """Run all decoupled brain tests."""
    print("🧠 Testing Decoupled Brain System")
    print("=" * 50)
    
    try:
        test_sensor_buffer()
        test_decoupled_brain_loop()
        test_concurrent_operation()
        
        print("\n🎉 All decoupled brain tests passed!")
        print("✅ Sensor buffer working correctly")
        print("✅ Brain loop running independently") 
        print("✅ 50ms cycle timing achievable")
        print("✅ Concurrent operation handling works")
        
        print("\n📋 Next Steps:")
        print("   • Integrate brain loop with TCP server")
        print("   • Add prediction queuing for clients")
        print("   • Implement proactive command generation")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)