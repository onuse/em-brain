#!/usr/bin/env python3
"""
Demo: How Telemetry Provides Real Insights

Shows the difference between heuristic-based testing and telemetry-based testing.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add paths
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))
testing_path = Path(__file__).parent.parent / 'testing'
sys.path.insert(0, str(testing_path))

from behavioral_test_dynamic import DynamicBehavioralTestFramework
from src.core.telemetry_client import TelemetryClient


def demo_prediction_learning():
    """Demo prediction learning with telemetry insights"""
    print("🔍 Demo: Prediction Learning with Telemetry")
    print("=" * 60)
    
    # Create framework
    framework = DynamicBehavioralTestFramework(quiet_mode=True)
    framework.setup_virtual_robot()
    
    # Get telemetry client
    telemetry = TelemetryClient()
    telemetry.connect()
    
    # Wait for session
    session_id = telemetry.wait_for_session()
    if not session_id:
        print("❌ No session available")
        return
    
    print(f"\n📊 Monitoring session: {session_id}")
    
    # Present repeating pattern
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    
    print("\n🎯 Presenting repeating pattern to brain...")
    print("-" * 60)
    
    for cycle in range(30):
        # Process pattern
        motor_output = framework.connection_handler.handle_sensory_input(
            framework.client_id, pattern
        )
        
        # Get telemetry every 5 cycles
        if cycle % 5 == 0:
            snapshot = telemetry.get_session_telemetry(session_id)
            if snapshot:
                print(f"\nCycle {cycle}:")
                print(f"  Prediction confidence: {snapshot.confidence:.3f}")
                print(f"  Cognitive mode: {snapshot.mode}")
                print(f"  Field energy: {snapshot.energy:.6f}")
                print(f"  Phase state: {snapshot.phase}")
                
                # Show motor output
                print(f"  Motor output: [{motor_output[0]:.3f}, {motor_output[1]:.3f}, ...]")
                
                # Insight that heuristics miss
                if snapshot.prediction_error:
                    print(f"  Prediction error: {snapshot.prediction_error:.3f}")
                
                if len(snapshot.prediction_history) > 0:
                    print(f"  Recent confidence: {[f'{c:.3f}' for c in snapshot.prediction_history[-3:]]}")
    
    # Now test with random pattern
    print("\n\n🎲 Switching to random pattern...")
    print("-" * 60)
    
    for cycle in range(10):
        # Random pattern
        random_pattern = list(np.random.rand(16))
        
        motor_output = framework.connection_handler.handle_sensory_input(
            framework.client_id, random_pattern
        )
        
        if cycle % 3 == 0:
            snapshot = telemetry.get_session_telemetry(session_id)
            if snapshot:
                print(f"\nCycle {cycle}:")
                print(f"  Prediction confidence: {snapshot.confidence:.3f}")
                print(f"  Cognitive mode: {snapshot.mode}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 INSIGHTS FROM TELEMETRY")
    print("=" * 60)
    
    final_snapshot = telemetry.get_session_telemetry(session_id)
    if final_snapshot:
        print(f"\nFinal state:")
        print(f"  Brain cycles: {final_snapshot.cycles}")
        print(f"  Memory regions: {final_snapshot.memory_regions}")
        print(f"  Active constraints: {final_snapshot.constraints}")
        print(f"  Improvement rate: {final_snapshot.improvement_rate:.3f}")
        
        print("\n✅ Telemetry reveals:")
        print("  - Actual prediction confidence (not motor variance)")
        print("  - Cognitive mode changes (exploration vs exploitation)")
        print("  - Field dynamics (energy, phase transitions)")
        print("  - Memory formation (regions, constraints)")
        print("  - True learning progress (improvement rate)")
    
    # Cleanup
    telemetry.disconnect()
    framework.cleanup()


def compare_approaches():
    """Compare heuristic vs telemetry approaches"""
    print("\n\n🔬 Comparing Testing Approaches")
    print("=" * 60)
    
    print("\n❌ Old Heuristic Approach:")
    print("  - Measured motor output variance")
    print("  - Assumed less variance = better prediction")
    print("  - No visibility into brain state")
    print("  - Could miss actual learning")
    
    print("\n✅ New Telemetry Approach:")
    print("  - Direct access to prediction confidence")
    print("  - Monitors cognitive modes and phases")
    print("  - Tracks memory and constraint formation")
    print("  - Measures actual improvement rate")
    print("  - Provides complete brain state visibility")


def main():
    """Run telemetry demo"""
    demo_prediction_learning()
    compare_approaches()
    
    print("\n\n🎉 Telemetry provides deep insights into brain operation!")
    print("   Tests can now verify actual brain behavior, not just outputs.")


if __name__ == "__main__":
    main()