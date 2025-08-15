#!/usr/bin/env python3
"""
Test telemetry system to debug confidence issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'server'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import time
from src.communication import MinimalBrainClient
from src.communication.monitoring_client import create_monitoring_client
from validation.embodied_learning.experiments.enhanced_telemetry_adapter import EnhancedTelemetryAdapter

def test_telemetry():
    """Test telemetry retrieval from brain server."""
    print("\n=== Testing Telemetry System ===\n")
    
    # Connect to brain server
    print("Connecting to brain server...")
    brain_client = MinimalBrainClient()
    if not brain_client.connect():
        print("Failed to connect to brain server. Is it running?")
        return
    
    print("✅ Connected to brain server")
    
    # Connect monitoring client
    print("Connecting to monitoring server...")
    monitoring_client = create_monitoring_client()
    if not monitoring_client:
        print("Failed to connect to monitoring server")
        brain_client.disconnect()
        return
        
    print("✅ Connected to monitoring server")
    
    # Create telemetry adapter
    telemetry_adapter = EnhancedTelemetryAdapter(monitoring_client)
    
    # Send some sensory data and get telemetry
    print("\nSending test data and retrieving telemetry...")
    
    for i in range(5):
        # Send sensory data
        sensory_data = [0.5] * 24  # Brain expects 24 sensors
        motor_response = brain_client.get_action(sensory_data)
        print(f"\nCycle {i+1}:")
        print(f"  Motor response: {motor_response[:3]}... (truncated)")
        
        # Get telemetry
        telemetry = telemetry_adapter.get_evolved_telemetry()
        if telemetry:
            print(f"  Telemetry keys: {list(telemetry.keys())[:5]}... (showing first 5)")
            
            # Check for prediction_confidence
            if 'prediction_confidence' in telemetry:
                print(f"  ✅ prediction_confidence: {telemetry['prediction_confidence']}")
            else:
                print(f"  ❌ prediction_confidence NOT FOUND in telemetry")
                
            # Check other relevant fields
            if 'evolution_state' in telemetry:
                evo = telemetry['evolution_state']
                print(f"  Evolution state: self_mod={evo.get('self_modification_strength', 'N/A')}")
                
            if 'predictive_phases' in telemetry:
                phases = telemetry['predictive_phases']
                print(f"  Predictive phases: {phases}")
        else:
            print("  ❌ No telemetry received")
        
        time.sleep(0.1)
    
    # Clean up
    brain_client.disconnect()
    monitoring_client.disconnect()
    print("\nTest complete!")

if __name__ == "__main__":
    test_telemetry()