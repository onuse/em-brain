#!/usr/bin/env python3
"""
Debug Field Stabilization Issue

The field stabilization test is failing with a score of 0.0.
This script investigates why field energy doesn't stabilize.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.direct_telemetry import DirectTelemetry


def debug_field_stabilization():
    """Debug why field stabilization fails"""
    print("🔍 Debugging Field Stabilization")
    print("=" * 60)
    
    # Initialize components
    robot_registry = RobotRegistry()
    brain_config = {
        'quiet_mode': False,
        'spatial_resolution': 4
    }
    brain_factory = DynamicBrainFactory(brain_config)
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    brain_service = BrainService(brain_pool, adapter_factory)
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    # Use direct telemetry (no sockets needed)
    telemetry_client = DirectTelemetry(
        brain_service=brain_service,
        connection_handler=connection_handler
    )
    
    # Setup robot
    client_id = "debug_robot"
    capabilities = [1.0, 16.0, 4.0, 0.0, 0.0]
    response = connection_handler.handle_handshake(client_id, capabilities)
    
    # Wait for session
    session_id = telemetry_client.wait_for_session(max_wait=2.0, client_id=client_id)
    print(f"\n📍 Session established: {session_id}")
    
    # Run field stabilization test
    print("\n🧪 Running Field Stabilization Test")
    print("-" * 40)
    
    stable_input = [0.5] * 16
    energy_levels = []
    confidence_levels = []
    timestamps = []
    
    # Collect more data for analysis
    cycles = 100
    print(f"Feeding constant input [0.5]*16 for {cycles} cycles...")
    
    start_time = time.time()
    
    for i in range(cycles):
        # Process input
        motor_output = connection_handler.handle_sensory_input(client_id, stable_input)
        
        # Get telemetry
        telemetry = telemetry_client.get_session_telemetry(session_id)
        if telemetry:
            energy_levels.append(telemetry.energy)
            confidence_levels.append(telemetry.confidence)
            timestamps.append(time.time() - start_time)
            
            # Print progress
            if i % 10 == 0:
                print(f"  Cycle {i}: energy={telemetry.energy:.6f}, "
                      f"confidence={telemetry.confidence:.3f}")
    
    print(f"\nCollected {len(energy_levels)} energy readings")
    
    # Analyze results
    print("\n📊 Analysis")
    print("-" * 40)
    
    if len(energy_levels) >= 20:
        # Calculate variances
        early_variance = np.var(energy_levels[:10])
        late_variance = np.var(energy_levels[-10:])
        
        print(f"Early variance (first 10): {early_variance:.6f}")
        print(f"Late variance (last 10): {late_variance:.6f}")
        
        if early_variance > 0:
            stability_improvement = 1.0 - (late_variance / early_variance)
            score = max(0, min(1.0, stability_improvement))
            print(f"Stability improvement: {stability_improvement:.3f}")
            print(f"Score: {score:.3f}")
        else:
            print("⚠️  Early variance is 0! This is why the test fails.")
            print("   The field starts already stable, so no improvement is possible.")
        
        # Additional statistics
        print(f"\nEnergy statistics:")
        print(f"  Mean: {np.mean(energy_levels):.6f}")
        print(f"  Std: {np.std(energy_levels):.6f}")
        print(f"  Min: {np.min(energy_levels):.6f}")
        print(f"  Max: {np.max(energy_levels):.6f}")
        print(f"  Range: {np.max(energy_levels) - np.min(energy_levels):.6f}")
        
        # Check if energy is actually changing
        unique_energies = len(set(energy_levels))
        print(f"  Unique values: {unique_energies}")
        
        if unique_energies == 1:
            print("\n❌ PROBLEM: Energy never changes! Field is stuck.")
        elif np.std(energy_levels) < 1e-6:
            print("\n❌ PROBLEM: Energy changes are too small to measure.")
        
        # Plot if available
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, energy_levels, 'b-', label='Field Energy')
            plt.axhline(y=np.mean(energy_levels), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(energy_levels):.6f}')
            plt.xlabel('Time (s)')
            plt.ylabel('Field Energy')
            plt.title('Field Energy Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, confidence_levels, 'g-', label='Prediction Confidence')
            plt.xlabel('Time (s)')
            plt.ylabel('Confidence')
            plt.title('Prediction Confidence Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('field_stabilization_debug.png', dpi=150)
            print("\n📈 Saved plot to field_stabilization_debug.png")
        except Exception as e:
            print(f"\n⚠️  Could not create plot: {e}")
    
    # Cleanup
    connection_handler.handle_disconnect(client_id)
    telemetry_client.disconnect()
    
    print("\n✅ Debug complete")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("1. Field may start too stable (variance = 0)")
    print("2. Consider initializing with some noise/instability")
    print("3. Or change test to measure absolute stability, not improvement")
    print("4. Could also test response to perturbations instead")


if __name__ == "__main__":
    debug_field_stabilization()