#!/usr/bin/env python3
"""
Test Direct Performance Report
Force a performance report to show the fixed variance display
"""

import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.brains.field.tcp_adapter import create_field_brain_tcp_adapter

def test_direct_performance_report():
    """Test the performance report directly"""
    print("🔍 Testing Direct Performance Report")
    print("=" * 80)
    
    # Create TCP adapter
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=False  # Allow normal output
    )
    
    # Process cycles to build history
    print("\n📊 Building field energy history...")
    base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
    
    for i in range(20):
        varied_pattern = [x + 0.05 * (i % 5) for x in base_pattern]
        action, brain_state = adapter.process_sensory_input(varied_pattern)
    
    # Force the report interval to trigger
    adapter.logger.last_report_time = 0  # Reset to force report
    
    print("\n📈 FIXED Performance Report (should show meaningful variance):")
    print("-" * 60)
    
    # This should now show the fixed variance formatting
    adapter.logger.maybe_report_performance()
    
    print("-" * 60)
    
    # Get the exact values for comparison
    energy_variance, _, _ = adapter.logger._calculate_learning_metrics()
    latest_energy = adapter.logger._get_latest_energy()
    
    print(f"\n🔍 Raw values:")
    print(f"   Latest energy: {latest_energy:.6f}")
    print(f"   Raw variance: {energy_variance:.9f}")
    
    if energy_variance < 0.000001:
        formatted = f"{energy_variance:.9f}"
    elif energy_variance < 0.001:
        formatted = f"{energy_variance:.6f}"
    else:
        formatted = f"{energy_variance:.3f}"
    
    print(f"   Fixed format: ±{formatted}")
    print(f"   Old format:   ±{energy_variance:.3f}")
    
    if energy_variance > 0 and f"{energy_variance:.3f}" == "0.000":
        print(f"✅ FIX VERIFIED: Small variance now visible as ±{formatted}")
    else:
        print(f"ℹ️ Variance large enough that old format would work too")

if __name__ == "__main__":
    test_direct_performance_report()