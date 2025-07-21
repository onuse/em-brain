#!/usr/bin/env python3
"""
Test TCP Field Evolution Detection

Test whether the TCP adapter now properly detects field evolution events
and reports them in performance logs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brains.field.tcp_adapter import create_field_brain_tcp_adapter
import time

def test_tcp_evolution_detection():
    """
    Test TCP adapter field evolution detection.
    """
    print("🔬 TCP Field Evolution Detection Test")
    print("=" * 45)
    
    # Create TCP adapter (this simulates production environment)
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=8,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=False
    )
    
    print(f"\n📊 Testing field evolution detection:")
    print(f"   Initial evolution cycles: {adapter.last_field_evolution_cycles}")
    
    # Process several inputs to trigger field evolution
    for i in range(5):
        sensory_input = [0.1 + i * 0.1] * 8
        action, brain_state = adapter.process_sensory_input(sensory_input)
        
        evolution_cycles = brain_state.get('field_evolution_cycles', 0)
        print(f"   Cycle {i+1}: evolution_cycles={evolution_cycles}, last_tracked={adapter.last_field_evolution_cycles}")
        
        # Small delay between cycles
        time.sleep(0.1)
    
    # Get performance stats from field logger
    print(f"\n📈 Field Logger Performance Report:")
    print(f"   TCP adapter tracking: {adapter.last_field_evolution_cycles} cycles")
    
    # Check the field logger's metrics directly
    logger_metrics = adapter.logger.metrics
    print(f"   Field logger evolution cycles: {logger_metrics.field_evolution_cycles}")
    
    # Force a performance report
    adapter.logger.maybe_report_performance()
    
    # Verify the fix worked
    if logger_metrics.field_evolution_cycles > 0:
        print(f"\n✅ SUCCESS: TCP adapter properly detects field evolution!")
        print(f"   Field logger tracked: {logger_metrics.field_evolution_cycles} evolution events")
        return True
    else:
        print(f"\n❌ FAILURE: Field logger still shows 0 evolution cycles")
        return False

if __name__ == "__main__":
    success = test_tcp_evolution_detection()
    exit(0 if success else 1)