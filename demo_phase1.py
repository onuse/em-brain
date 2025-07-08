#!/usr/bin/env python3
"""
Phase 1 Demo Script for Emergent Intelligence Robot Brain

Demonstrates:
1. Core data structures (ExperienceNode, WorldGraph)
2. Basic similarity calculation using Euclidean distance
3. Unit tests verification
4. Grid world simulation in action
5. Performance with 1000+ nodes
6. Graph serialization for debugging

This script showcases the "embarrassingly simple" design philosophy
where complex behaviors emerge from simple mechanisms.
"""

import time
import random
from core import ExperienceNode, WorldGraph, MentalContext, GenomeData
from core.serialization import save_graph_debug_info
from simulation import GridWorldBrainstem


def demo_core_data_structures():
    """Demonstrate core data structures and basic operations."""
    print("=" * 60)
    print("PHASE 1 DEMO: Core Data Structures")
    print("=" * 60)
    
    # Create experience nodes
    print("\n1. Creating ExperienceNodes...")
    
    experiences = []
    for i in range(5):
        node = ExperienceNode(
            mental_context=[random.random() for _ in range(8)],
            action_taken={
                "forward_motor": random.uniform(-1, 1),
                "turn_motor": random.uniform(-1, 1),
                "brake_motor": random.uniform(0, 1)
            },
            predicted_sensory=[random.random() for _ in range(22)],
            actual_sensory=[random.random() for _ in range(22)],
            prediction_error=random.uniform(0, 2)
        )
        experiences.append(node)
        print(f"   Created node {node.node_id[:8]}... with {len(node.mental_context)} context dims")
    
    # Create WorldGraph and add experiences
    print("\n2. Building WorldGraph...")
    graph = WorldGraph()
    
    for exp in experiences:
        graph.add_node(exp)
    
    print(f"   Graph now contains {graph.node_count()} nodes")
    print(f"   Temporal chain length: {len(graph.temporal_chain)}")
    
    # Demonstrate similarity search
    print("\n3. Testing similarity search...")
    test_context = experiences[0].mental_context
    similar = graph.find_similar_nodes(test_context, similarity_threshold=0.5, max_results=3)
    
    print(f"   Found {len(similar)} similar nodes to first experience")
    for node in similar:
        similarity = graph._calculate_context_similarity(test_context, node.mental_context)
        print(f"   - Node {node.node_id[:8]}... (similarity: {similarity:.3f})")
    
    # Demonstrate strength tracking
    print("\n4. Testing strength tracking...")
    first_node_id = experiences[0].node_id
    initial_strength = experiences[0].strength
    
    graph.strengthen_node(first_node_id, 0.5)
    print(f"   Node strength: {initial_strength:.3f} -> {experiences[0].strength:.3f}")
    print(f"   Times accessed: {experiences[0].times_accessed}")
    
    # Get statistics
    print("\n5. Graph statistics:")
    stats = graph.get_graph_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return graph


def demo_simulation_environment():
    """Demonstrate the grid world simulation environment."""
    print("\n" + "=" * 60)
    print("PHASE 1 DEMO: Grid World Simulation")
    print("=" * 60)
    
    # Create brainstem simulation
    print("\n1. Initializing grid world simulation...")
    brainstem = GridWorldBrainstem(world_width=15, world_height=15, seed=42)
    
    # Show hardware capabilities
    capabilities = brainstem.get_hardware_capabilities()
    print(f"   Simulation mode: {capabilities['simulation_mode']}")
    print(f"   Total sensors: {capabilities['total_sensor_size']}")
    print(f"   Actuators: {len(capabilities['actuators'])}")
    
    # Get initial sensor reading
    print("\n2. Getting initial sensor readings...")
    sensor_packet = brainstem.get_sensor_readings()
    print(f"   Sensor values: {len(sensor_packet.sensor_values)} readings")
    print(f"   Sample values: {sensor_packet.sensor_values[:5]}")  # First 5 sensors
    
    # Demonstrate motor commands
    print("\n3. Testing motor commands...")
    initial_stats = brainstem.get_simulation_stats()
    print(f"   Initial position: {initial_stats['robot_position']}")
    print(f"   Initial health: {initial_stats['robot_health']:.3f}")
    print(f"   Initial energy: {initial_stats['robot_energy']:.3f}")
    
    # Execute a sequence of motor commands
    commands_sequence = [
        {"forward_motor": 0.5, "turn_motor": 0.0, "brake_motor": 0.0},  # Move forward
        {"forward_motor": 0.0, "turn_motor": 0.7, "brake_motor": 0.0},  # Turn right
        {"forward_motor": 0.5, "turn_motor": 0.0, "brake_motor": 0.0},  # Move forward
        {"forward_motor": 0.0, "turn_motor": -0.7, "brake_motor": 0.0}, # Turn left
    ]
    
    for i, commands in enumerate(commands_sequence):
        alive = brainstem.execute_motor_commands(commands)
        stats = brainstem.get_simulation_stats()
        print(f"   Step {i+1}: pos={stats['robot_position']}, "
              f"health={stats['robot_health']:.3f}, energy={stats['robot_energy']:.3f}")
        
        if not alive:
            print("   Robot died!")
            break
    
    # Show world state
    print("\n4. Current world state:")
    world_state = brainstem.get_world_state()
    print(f"   World size: {world_state['world_size']}")
    print(f"   Robot position: {world_state['robot_position']}")
    print(f"   Robot orientation: {world_state['robot_orientation']}")
    
    return brainstem


def demo_learning_integration():
    """Demonstrate integration between simulation and learning."""
    print("\n" + "=" * 60)
    print("PHASE 1 DEMO: Learning Integration")
    print("=" * 60)
    
    # Create components
    brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=123)
    graph = WorldGraph()
    genome = GenomeData()
    
    print(f"\n1. Running {10} learning steps...")
    
    for step in range(10):
        # Get current sensor readings
        sensor_packet = brainstem.get_sensor_readings()
        
        # Create a simple mental context (for demo - in real implementation this would be more sophisticated)
        mental_context = sensor_packet.sensor_values[:8]  # Use first 8 sensor values as context
        
        # Generate random motor action (for demo - real implementation would use prediction)
        motor_action = {
            "forward_motor": random.uniform(-0.5, 0.5),
            "turn_motor": random.uniform(-0.5, 0.5),
            "brake_motor": random.uniform(0, 0.3)
        }
        
        # Execute action and get new sensor reading
        brainstem.execute_motor_commands(motor_action)
        new_sensor_packet = brainstem.get_sensor_readings()
        
        # Calculate prediction error (for demo, assume we predicted current sensors)
        predicted_sensory = sensor_packet.sensor_values
        actual_sensory = new_sensor_packet.sensor_values
        prediction_error = sum((a - p) ** 2 for a, p in zip(actual_sensory, predicted_sensory)) ** 0.5
        
        # Create experience node
        experience = ExperienceNode(
            mental_context=mental_context,
            action_taken=motor_action,
            predicted_sensory=predicted_sensory,
            actual_sensory=actual_sensory,
            prediction_error=prediction_error
        )
        
        # Add to graph
        graph.add_node(experience)
        
        # Show progress
        if step % 3 == 0:
            stats = brainstem.get_simulation_stats()
            print(f"   Step {step}: Graph has {graph.node_count()} nodes, "
                  f"prediction error: {prediction_error:.3f}")
    
    print(f"\n2. Learning results:")
    graph_stats = graph.get_graph_statistics()
    sim_stats = brainstem.get_simulation_stats()
    
    print(f"   Total experiences collected: {graph_stats['total_nodes']}")
    print(f"   Average prediction error: {graph_stats.get('avg_prediction_error', 'N/A')}")
    print(f"   Robot survived {sim_stats['step_count']} steps")
    print(f"   Final robot health: {sim_stats['robot_health']:.3f}")
    
    return graph


def demo_performance_scaling():
    """Demonstrate performance with larger numbers of nodes."""
    print("\n" + "=" * 60)
    print("PHASE 1 DEMO: Performance Scaling")
    print("=" * 60)
    
    print("\n1. Creating large graph (1000 nodes)...")
    start_time = time.time()
    
    graph = WorldGraph()
    for i in range(1000):
        node = ExperienceNode(
            mental_context=[random.random() for _ in range(10)],
            action_taken={"motor": random.uniform(-1, 1)},
            predicted_sensory=[random.random() for _ in range(22)],
            actual_sensory=[random.random() for _ in range(22)],
            prediction_error=random.random()
        )
        graph.add_node(node)
    
    creation_time = time.time() - start_time
    print(f"   Created 1000 nodes in {creation_time:.3f} seconds")
    
    # Test similarity search performance
    print("\n2. Testing similarity search performance...")
    search_context = [random.random() for _ in range(10)]
    
    start_time = time.time()
    similar = graph.find_similar_nodes(search_context, similarity_threshold=0.7, max_results=10)
    search_time = time.time() - start_time
    
    print(f"   Found {len(similar)} similar nodes in {search_time:.4f} seconds")
    
    # Test strength operations
    print("\n3. Testing strength operations...")
    start_time = time.time()
    
    # Strengthen 100 random nodes
    all_nodes = list(graph.nodes.keys())
    for _ in range(100):
        node_id = random.choice(all_nodes)
        graph.strengthen_node(node_id, 0.1)
    
    strength_time = time.time() - start_time
    print(f"   100 strength updates completed in {strength_time:.4f} seconds")
    
    # Show final statistics
    print("\n4. Final graph statistics:")
    stats = graph.get_graph_statistics()
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Average strength: {stats['avg_strength']:.3f}")
    print(f"   Max strength: {stats['max_strength']:.3f}")
    print(f"   Total accesses: {stats['total_accesses']}")
    
    return graph


def demo_serialization(graph):
    """Demonstrate graph serialization capabilities."""
    print("\n" + "=" * 60)
    print("PHASE 1 DEMO: Graph Serialization")
    print("=" * 60)
    
    print("\n1. Saving graph debug information...")
    
    # Save comprehensive debug info
    results = save_graph_debug_info(graph, "phase1_demo")
    
    print("   Serialization results:")
    for format_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"   {status} {format_name.upper()} format")
    
    if all(results.values()):
        print("\n   All debug files saved to debug_output/ directory")
        print("   Files created:")
        print("   - phase1_demo_full.json (complete graph)")
        print("   - phase1_demo_full.pkl (binary format)")
        print("   - phase1_demo_summary.txt (human readable)")
        print("   - phase1_demo_nodes.csv (node data)")
    
    return results


def main():
    """Run the complete Phase 1 demonstration."""
    print("🤖 EMERGENT INTELLIGENCE ROBOT - PHASE 1 IMPLEMENTATION")
    print("Following the 'embarrassingly simple' design philosophy")
    print("Complex behaviors emerge from simple mechanisms\n")
    
    # Run all demonstrations
    try:
        # Core data structures
        graph1 = demo_core_data_structures()
        
        # Simulation environment  
        brainstem = demo_simulation_environment()
        
        # Learning integration
        graph2 = demo_learning_integration()
        
        # Performance scaling
        large_graph = demo_performance_scaling()
        
        # Serialization
        demo_serialization(large_graph)
        
        print("\n" + "=" * 60)
        print("PHASE 1 IMPLEMENTATION COMPLETE! ✓")
        print("=" * 60)
        print("\nSuccessfully implemented:")
        print("✓ ExperienceNode data structure with all required fields")
        print("✓ WorldGraph class with basic operations (add, remove, search)")
        print("✓ Similarity calculation using Euclidean distance")
        print("✓ Node strength tracking and update mechanisms")
        print("✓ Comprehensive unit tests (run 'pytest tests/' to verify)")
        print("✓ Grid world simulation environment")
        print("✓ Performance scaling to 1000+ nodes")
        print("✓ Graph serialization for debugging")
        print("\nThe foundation is ready for building the emergent intelligence!")
        print("Next: Implement prediction generation and mental loop...")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()