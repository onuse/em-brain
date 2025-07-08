#!/usr/bin/env python3
"""
Simple test script to verify predictor integration.
This tests the predictor system without GUI interaction.
"""

import random
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from predictor.triple_predictor import TriplePredictor
from visualization.brain_monitor import BrainStateMonitor


def create_sample_experiences(graph: WorldGraph, num_experiences: int = 15):
    """Create sample experiences for testing."""
    print(f"Creating {num_experiences} sample experiences...")
    
    for i in range(num_experiences):
        # Create diverse mental contexts
        mental_context = [
            random.uniform(-2.0, 2.0),  # position_x
            random.uniform(-2.0, 2.0),  # position_y
            random.uniform(0.0, 1.0),   # energy_level
            random.uniform(0.0, 1.0),   # health_level
            random.uniform(0.0, 1.0),   # food_nearby
        ]
        
        # Create diverse actions
        action_taken = {
            "forward_motor": random.uniform(-1.0, 1.0),
            "turn_motor": random.uniform(-1.0, 1.0),
            "brake_motor": random.uniform(0.0, 1.0)
        }
        
        # Create sensory data (simplified)
        sensory_data = [random.uniform(0.0, 1.0) for _ in range(22)]
        
        # Create experience node
        node = ExperienceNode(
            mental_context=mental_context,
            action_taken=action_taken,
            predicted_sensory=sensory_data,
            actual_sensory=[s + random.uniform(-0.1, 0.1) for s in sensory_data],
            prediction_error=random.uniform(0.0, 1.0),
            strength=random.uniform(0.5, 3.0)
        )
        
        graph.add_node(node)
    
    # Add some connections between similar nodes
    all_nodes = graph.all_nodes()
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            if random.random() < 0.3:  # 30% chance of connection
                if graph._calculate_context_similarity(
                    all_nodes[i].mental_context, 
                    all_nodes[j].mental_context
                ) > 0.7:
                    all_nodes[i].similar_contexts.append(all_nodes[j].node_id)
                    all_nodes[j].similar_contexts.append(all_nodes[i].node_id)
    
    print(f"Created {len(all_nodes)} nodes with connections")


def main():
    """Main test function."""
    print("=== Simple Predictor Integration Test ===")
    
    # Initialize pygame for brain monitor
    import pygame
    pygame.init()
    
    # Initialize components
    world_graph = WorldGraph()
    predictor = TriplePredictor(max_depth=3, randomness_factor=0.3, base_time_budget=0.02)
    
    # Create sample experiences
    create_sample_experiences(world_graph, 15)
    
    # Test scenarios with different threat levels
    test_scenarios = [
        ([0.0, 0.0, 0.8, 0.9, 0.1], "safe"),      # High energy, low food - safe environment
        ([1.0, 1.0, 0.3, 0.7, 0.8], "normal"),    # Low energy, high food - normal
        ([-1.0, -1.0, 0.6, 0.5, 0.0], "alert"),   # Medium energy, no food - alert
        ([0.5, -0.5, 0.9, 0.8, 0.5], "danger"),   # High energy, medium food - danger
    ]
    
    print(f"Testing {len(test_scenarios)} different scenarios...")
    
    for i, (context, threat_level) in enumerate(test_scenarios):
        print(f"\n--- Test {i+1}: Context {context}, Threat: {threat_level} ---")
        
        # Generate prediction with threat level
        consensus_result = predictor.generate_prediction(
            context, world_graph, i, threat_level
        )
        
        print(f"Consensus strength: {consensus_result.consensus_strength}")
        print(f"Agreement: {consensus_result.agreement_count}/{consensus_result.total_traversals}")
        print(f"Reasoning: {consensus_result.reasoning}")
        print(f"Action: {consensus_result.prediction.motor_action}")
        print(f"Confidence: {consensus_result.prediction.confidence:.3f}")
        print(f"Traversals completed: {consensus_result.prediction.traversal_count}")
        print(f"Thinking time used: {consensus_result.prediction.time_budget_used:.4f}s")
        print(f"Threat level: {consensus_result.prediction.threat_level}")
        
        # Simulate prediction error
        prediction_error = random.uniform(0.0, 2.0)
        
        # Test brain monitor update (without rendering)
        brain_monitor = BrainStateMonitor(400, 600)
        brain_monitor.update(
            world_graph, 
            prediction_error, 
            consensus_result.prediction.motor_action,
            i,
            consensus_result
        )
        
        print(f"Brain monitor updated successfully")
    
    # Print final statistics
    print("\n=== Final Predictor Statistics ===")
    predictor_stats = predictor.get_prediction_statistics()
    for key, value in predictor_stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Graph Statistics ===")
    graph_stats = world_graph.get_graph_statistics()
    for key, value in graph_stats.items():
        print(f"{key}: {value}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()