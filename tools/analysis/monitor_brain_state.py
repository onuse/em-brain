#!/usr/bin/env python3
"""Monitor real-time brain state to debug telemetry issues."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import time
from src.communication.monitoring_client import BrainMonitoringClient

print("Connecting to brain monitoring server...")
monitor = BrainMonitoringClient('localhost', 9998)
monitor.connect()

print("\nMonitoring brain state (press Ctrl+C to stop)...")
print("-" * 80)

try:
    for i in range(20):
        telemetry = monitor.get_telemetry()
        
        if telemetry:
            print(f"\nCycle {i}:")
            print(f"  Raw telemetry keys: {list(telemetry.keys())[:10]}...")
            
            # Check for both old and new field names
            field_info = telemetry.get('field_information', telemetry.get('field_energy', -1))
            print(f"  Field information/energy: {field_info}")
            
            # Check confidence
            confidence = telemetry.get('prediction_confidence', -1)
            print(f"  Prediction confidence: {confidence}")
            
            # Check evolution state
            evo_state = telemetry.get('evolution_state', {})
            if evo_state:
                print(f"  Self-modification: {evo_state.get('self_modification_strength', 0):.1%}")
                print(f"  Smoothed information: {evo_state.get('smoothed_information', evo_state.get('smoothed_energy', 0)):.3f}")
                print(f"  Smoothed confidence: {evo_state.get('smoothed_confidence', 0):.3f}")
            
            # Check information state
            info_state = telemetry.get('information_state', telemetry.get('energy_state', {}))
            if info_state:
                print(f"  Information level: {info_state.get('information', info_state.get('energy', 0)):.3f}")
                print(f"  Exploration drive: {info_state.get('exploration_drive', 0):.3f}")
            
        else:
            print(f"\nCycle {i}: No telemetry received")
        
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\nStopping monitor...")
finally:
    monitor.disconnect()
    print("Disconnected.")