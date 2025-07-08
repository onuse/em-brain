#!/usr/bin/env python3
"""
Demonstration of distributed brain architecture.
Shows brain server running on "laptop" with brainstem clients connecting via sockets.
"""

import asyncio
import time
from datetime import datetime
from network.brain_server import BrainSocketServer
from network.brain_client import BrainSocketClient
from simulation.brainstem_sim import GridWorldBrainstem
from core.communication import SensoryPacket


async def run_brain_server():
    """Run the brain server (simulates laptop RTX 3070)."""
    print("🧠 Starting Brain Server (simulating laptop RTX 3070)...")
    
    server = BrainSocketServer(
        host="localhost", 
        port=8080, 
        base_time_budget=0.1
    )
    
    # Run server in background task
    server_task = asyncio.create_task(server.start_server())
    
    # Give server time to start
    await asyncio.sleep(1)
    print("🧠 Brain Server running on localhost:8080")
    
    return server_task, server


async def test_simple_client():
    """Test a simple brainstem client connection."""
    print("\n🤖 Testing Simple Brainstem Client...")
    
    client = BrainSocketClient(client_name="test_brainstem")
    
    try:
        # Connect to brain
        if await client.connect():
            print("✅ Connected to brain server")
            
            # Send some test sensory data
            for i in range(3):
                sensory = SensoryPacket(
                    sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0],  # 5 sensors
                    actuator_positions=[0.0, 0.0],
                    timestamp=datetime.now(),
                    sequence_id=i + 1
                )
                
                mental_context = [float(i), 0.5, 0.8]  # position, energy, health
                
                prediction = await client.process_sensory_input(
                    sensory, mental_context, "normal"
                )
                
                if prediction:
                    print(f"  Step {i+1}: Brain responded with {prediction.motor_action}")
                    print(f"    Consensus: {prediction.consensus_strength}, "
                          f"Traversals: {prediction.traversal_count}")
                else:
                    print(f"  Step {i+1}: No response from brain")
                
                await asyncio.sleep(0.1)
                
            print("✅ Simple client test completed")
        else:
            print("❌ Failed to connect to brain server")
            
    finally:
        await client.disconnect()


async def test_grid_world_brainstem():
    """Test GridWorld brainstem using socket communication."""
    print("\n🌍 Testing GridWorld Brainstem (simulating Pi Zero 2 WH)...")
    
    # Create grid world that connects to brain server
    brainstem = GridWorldBrainstem(
        world_width=10, 
        world_height=10,
        brain_host="localhost",
        brain_port=8080,
        use_sockets=True  # Use socket communication
    )
    
    try:
        # Run brain-controlled simulation
        results = await brainstem.run_brain_controlled_simulation(
            steps=20, 
            step_delay=0.05  # Fast simulation
        )
        
        print(f"✅ GridWorld simulation completed:")
        print(f"   Steps: {results['steps_completed']}")
        print(f"   Predictions: {results['predictions_received']}")
        print(f"   Errors: {results['communication_errors']}")
        print(f"   Final robot health: {results['final_robot_state']['health']:.2f}")
        print(f"   Final robot energy: {results['final_robot_state']['energy']:.2f}")
        
        # Show some performance stats
        if results['performance_stats']:
            last_stats = results['performance_stats'][-1]
            print(f"   Final consensus: {last_stats['consensus_strength']}")
            print(f"   Final traversals: {last_stats['traversal_count']}")
        
    except Exception as e:
        print(f"❌ GridWorld test failed: {e}")


async def test_multiple_brainstems():
    """Test multiple brainstem clients connecting simultaneously."""
    print("\n🔗 Testing Multiple Brainstem Connections...")
    
    # Create multiple brainstem clients
    clients = []
    for i in range(3):
        client = BrainSocketClient(client_name=f"brainstem_{i+1}")
        clients.append(client)
    
    try:
        # Connect all clients
        connected_clients = []
        for client in clients:
            if await client.connect():
                connected_clients.append(client)
                print(f"✅ {client.client_name} connected")
        
        print(f"🔗 {len(connected_clients)} brainstems connected to brain")
        
        # Send requests from all clients simultaneously
        tasks = []
        for i, client in enumerate(connected_clients):
            sensory = SensoryPacket(
                sensor_values=[float(i), float(i+1), float(i+2)],  # Different sensor configs
                actuator_positions=[0.0],
                timestamp=datetime.now(),
                sequence_id=1
            )
            
            task = client.process_sensory_input(
                sensory, [float(i)], "normal"
            )
            tasks.append(task)
        
        # Wait for all responses
        predictions = await asyncio.gather(*tasks)
        
        for i, prediction in enumerate(predictions):
            if prediction:
                print(f"  Brainstem {i+1}: Got prediction with {prediction.traversal_count} traversals")
            else:
                print(f"  Brainstem {i+1}: No response")
        
        print("✅ Multiple brainstem test completed")
        
    finally:
        # Disconnect all clients
        for client in connected_clients:
            await client.disconnect()


async def test_brain_statistics():
    """Test brain statistics retrieval."""
    print("\n📊 Testing Brain Statistics...")
    
    client = BrainSocketClient(client_name="stats_client")
    
    try:
        if await client.connect():
            stats = await client.get_brain_statistics()
            
            if stats:
                print("✅ Retrieved brain statistics:")
                print(f"   Sensory vector length: {stats['interface_stats']['sensory_vector_length']}")
                print(f"   Total experiences: {stats['interface_stats']['total_experiences']}")
                print(f"   Total predictions: {stats['predictor_stats']['total_predictions']}")
                print(f"   Connected clients: {stats['server_stats']['connected_clients']}")
                print(f"   Strong consensus rate: {stats['predictor_stats']['strong_consensus_rate']:.1f}%")
            else:
                print("❌ Failed to get brain statistics")
        
    finally:
        await client.disconnect()


async def main():
    """Main demonstration function."""
    print("=== Distributed Brain Architecture Demo ===")
    print("Brain Server: Laptop (RTX 3070)")
    print("Brainstem Clients: Pi Zero 2 WH + Simulations")
    print("Communication: WebSockets over local network")
    print()
    
    # Start brain server
    server_task, server = await run_brain_server()
    
    try:
        # Run various tests
        await test_simple_client()
        await test_grid_world_brainstem()
        await test_multiple_brainstems()
        await test_brain_statistics()
        
        print("\n🎉 All distributed brain tests passed!")
        
        # Show final server statistics
        server_info = server.get_server_info()
        print(f"\n📊 Final Brain Server Stats:")
        print(f"   Total predictions served: {server_info['total_predictions']}")
        print(f"   Current connections: {server_info['connected_clients']}")
        print(f"   Brain experiences: {server_info['brain_stats']['interface_stats']['total_experiences']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    finally:
        # Cleanup
        print("\n🧹 Shutting down brain server...")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        print("✅ Demo completed")


if __name__ == "__main__":
    # Install websockets if not available
    try:
        import websockets
    except ImportError:
        print("Installing websockets...")
        import subprocess
        subprocess.check_call(["pip", "install", "websockets"])
        import websockets
    
    asyncio.run(main())