#!/usr/bin/env python3
"""
Simple test of socket communication between brain server and client.
"""

import asyncio
import json
from datetime import datetime
from network.brain_server import BrainSocketServer
from network.brain_client import BrainSocketClient
from core.communication import SensoryPacket


async def test_basic_socket_communication():
    """Test basic socket communication without long-running server."""
    print("🧠 Testing Basic Socket Communication...")
    
    # Start brain server in background
    server = BrainSocketServer(host="localhost", port=8081, base_time_budget=0.01)
    
    # Create server task but don't await it
    server_task = asyncio.create_task(server.start_server())
    
    # Give server time to start
    await asyncio.sleep(0.5)
    
    try:
        print("✅ Brain server started")
        
        # Create client and test communication
        client = BrainSocketClient(brain_host="localhost", brain_port=8081, client_name="test")
        
        # Test connection
        if await client.connect():
            print("✅ Client connected to brain server")
            
            # Test sensory input processing
            sensory = SensoryPacket(
                sensor_values=[1.0, 2.0, 3.0, 4.0],
                actuator_positions=[0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=1
            )
            
            prediction = await client.process_sensory_input(
                sensory, [0.0, 1.0], "normal"
            )
            
            if prediction:
                print(f"✅ Received prediction: {prediction.motor_action}")
                print(f"   Consensus: {prediction.consensus_strength}")
                print(f"   Traversals: {prediction.traversal_count}")
                print(f"   Sensory length: {len(prediction.expected_sensory)}")
            else:
                print("❌ No prediction received")
            
            # Test brain statistics
            stats = await client.get_brain_statistics()
            if stats:
                print(f"✅ Brain stats: {stats['interface_stats']['sensory_vector_length']} sensors learned")
            
            await client.disconnect()
            print("✅ Client disconnected")
            
        else:
            print("❌ Failed to connect client")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        # Stop server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        print("✅ Brain server stopped")


async def test_json_serialization():
    """Test JSON serialization of communication packets."""
    print("\n📦 Testing JSON Serialization...")
    
    # Test SensoryPacket serialization
    sensory = SensoryPacket(
        sensor_values=[1.0, 2.0, 3.0],
        actuator_positions=[0.1, 0.2],
        timestamp=datetime.now(),
        sequence_id=42,
        network_latency=0.001
    )
    
    # Serialize and deserialize
    json_str = sensory.to_json()
    sensory_copy = SensoryPacket.from_json(json_str)
    
    print(f"✅ SensoryPacket: {len(sensory.sensor_values)} sensors serialized/deserialized")
    assert sensory.sensor_values == sensory_copy.sensor_values
    assert sensory.sequence_id == sensory_copy.sequence_id
    
    print("✅ JSON serialization test passed")


async def main():
    """Run basic socket communication tests."""
    print("=== Socket Communication Test ===")
    
    await test_json_serialization()
    await test_basic_socket_communication()
    
    print("\n🎉 All socket tests completed!")


if __name__ == "__main__":
    asyncio.run(main())