#!/usr/bin/env python3
"""
Inspect Running Brain State

Connect to a running brain server and inspect its current state,
field brain statistics, and learning progress.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)

from communication import MinimalBrainClient


def inspect_running_brain(host='localhost', port=9999):
    """Connect to running brain and inspect its state."""
    print(f"🔍 Inspecting running brain at {host}:{port}")
    
    try:
        # Connect to brain server
        client = MinimalBrainClient(host=host, port=port)
        client.connect()
        print("✅ Connected to brain server")
        
        # Send a minimal sensory input to get brain state
        test_input = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        action, brain_state = client.send_sensory_input(test_input)
        
        print(f"\n🧠 Brain State Analysis:")
        print(f"   Action response: {[f'{x:.3f}' for x in action]}")
        
        # Basic brain info
        if 'brain_cycles' in brain_state:
            print(f"   Brain cycles: {brain_state['brain_cycles']:,}")
        
        if 'prediction_confidence' in brain_state:
            print(f"   Prediction confidence: {brain_state['prediction_confidence']:.4f}")
        
        if 'field_energy' in brain_state:
            print(f"   Field energy: {brain_state['field_energy']:.3f}")
        
        if 'processing_mode' in brain_state:
            print(f"   Processing mode: {brain_state['processing_mode']}")
        
        # Field brain specific metrics
        if 'field_evolution_cycles' in brain_state:
            print(f"\n🔬 Field Brain Metrics:")
            print(f"   Evolution cycles: {brain_state.get('field_evolution_cycles', 0)}")
            print(f"   Topology regions: {brain_state.get('topology_regions', 0)}")
            print(f"   Field activation: {brain_state.get('field_activation', 0.0):.3f}")
        
        # Learning indicators
        if 'dynamics_families' in brain_state:
            families = brain_state['dynamics_families']
            print(f"\n⚡ Dynamics Activity:")
            for family, activity in families.items():
                print(f"   {family}: {activity:.3f}")
        
        # Enhanced features
        if 'enhanced_features' in brain_state:
            enhanced = brain_state['enhanced_features']
            print(f"\n🚀 Enhanced Features:")
            for feature, status in enhanced.items():
                print(f"   {feature}: {status}")
        
        # Performance info
        if 'cycle_time_ms' in brain_state:
            print(f"\n⏱️ Performance:")
            print(f"   Cycle time: {brain_state['cycle_time_ms']:.1f}ms")
        
        # Stream interface
        if 'stream_status' in brain_state:
            stream = brain_state['stream_status']
            print(f"\n📡 Stream Interface:")
            for key, value in stream.items():
                print(f"   {key}: {value}")
        
        # Show raw brain state (truncated)
        print(f"\n📊 Raw Brain State (sample):")
        sample_keys = ['prediction_method', 'brain_cycles', 'field_energy', 'prediction_confidence']
        for key in sample_keys:
            if key in brain_state:
                print(f"   {key}: {brain_state[key]}")
        
        # Disconnect
        client.disconnect()
        print(f"\n✅ Brain inspection completed")
        
    except ConnectionError as e:
        print(f"❌ Could not connect to brain server: {e}")
        print(f"   Make sure brain server is running on {host}:{port}")
    except Exception as e:
        print(f"❌ Error during brain inspection: {e}")


def main():
    """Main inspection function."""
    print("🔍 Running Brain Inspector")
    print("=" * 40)
    
    # Check default brain server
    inspect_running_brain()
    
    print("\nIf no brain server is running, start one with:")
    print("python3 field_brain_tcp_server.py")


if __name__ == "__main__":
    main()