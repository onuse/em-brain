#!/usr/bin/env python3
"""
Test confidence evolution over time with varying sensory input
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'server'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import time
import numpy as np
from src.communication import MinimalBrainClient
from src.communication.monitoring_client import create_monitoring_client
from validation.embodied_learning.experiments.enhanced_telemetry_adapter import EnhancedTelemetryAdapter

def test_confidence_evolution():
    """Test how confidence evolves with predictable sensory patterns."""
    print("\n=== Testing Confidence Evolution ===\n")
    
    # Connect to servers
    brain_client = MinimalBrainClient()
    if not brain_client.connect():
        print("Failed to connect to brain server")
        return
    
    monitoring_client = create_monitoring_client()
    if not monitoring_client:
        print("Failed to connect to monitoring server")
        brain_client.disconnect()
        return
        
    telemetry_adapter = EnhancedTelemetryAdapter(monitoring_client)
    
    print("Running test with predictable sensory patterns...\n")
    
    confidence_history = []
    
    # Run for 100 cycles with a predictable sine wave pattern
    for i in range(100):
        # Create predictable sensory pattern (sine wave)
        t = i * 0.1
        sensory_data = [0.5 + 0.3 * np.sin(t + j * 0.5) for j in range(24)]  # Brain expects 24 sensors
        
        # Send to brain
        motor_response = brain_client.get_action(sensory_data)
        
        # Get telemetry every 10 cycles
        if i % 10 == 0:
            telemetry = telemetry_adapter.get_evolved_telemetry()
            if telemetry:
                confidence = telemetry.get('prediction_confidence', 0.0)
                confidence_history.append(confidence)
                print(f"Cycle {i:3d}: confidence={confidence:.3f}")
                
                # Also check learning metrics and predictive phases
                if i == 50 or i == 90:
                    metrics = telemetry_adapter.get_learning_metrics()
                    print(f"  Learning metrics: efficiency={metrics['efficiency']:.3f}, " +
                          f"learning_detected={metrics['learning_detected']}")
                    
                    # Check predictive phases
                    if 'predictive_phases' in telemetry:
                        phases = telemetry['predictive_phases']
                        print(f"  Predictive phases: {phases}")
                    else:
                        print(f"  Predictive phases: NOT FOUND in telemetry")
        
        time.sleep(0.01)  # Small delay
    
    # Analyze confidence evolution
    print(f"\nConfidence evolution:")
    if confidence_history:
        print(f"  Initial: {confidence_history[0]:.3f}")
        print(f"  Final: {confidence_history[-1]:.3f}")
        print(f"  Change: {confidence_history[-1] - confidence_history[0]:+.3f}")
        print(f"  Max: {max(confidence_history):.3f}")
        print(f"  Min: {min(confidence_history):.3f}")
    else:
        print("  No confidence data collected - brain may be experiencing errors")
    
    # Clean up
    brain_client.disconnect()
    monitoring_client.disconnect()

if __name__ == "__main__":
    test_confidence_evolution()