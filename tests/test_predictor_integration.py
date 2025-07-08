#!/usr/bin/env python3
"""
Test script to demonstrate predictor integration with visualization.
This shows how the triple predictor works with the brain monitor.
"""

import pygame
import time
import random
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from predictor.triple_predictor import TriplePredictor
from visualization.brain_monitor import BrainStateMonitor


def create_sample_experiences(graph: WorldGraph, num_experiences: int = 20):
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
    print("=== Predictor Integration Test ===")
    
    # Initialize components
    world_graph = WorldGraph()
    predictor = TriplePredictor(max_depth=3, randomness_factor=0.3)
    
    # Create sample experiences
    create_sample_experiences(world_graph, 25)
    
    # Initialize pygame and brain monitor
    pygame.init()
    screen = pygame.display.set_mode((500, 700))
    pygame.display.set_caption("Predictor Integration Test")
    
    brain_monitor = BrainStateMonitor(500, 700)
    clock = pygame.time.Clock()
    
    # Test scenarios
    test_contexts = [
        [0.0, 0.0, 0.8, 0.9, 0.1],  # High energy, low food
        [1.0, 1.0, 0.3, 0.7, 0.8],  # Low energy, high food
        [-1.0, -1.0, 0.6, 0.5, 0.0],  # Medium energy, no food
        [0.5, -0.5, 0.9, 0.8, 0.5],  # High energy, medium food
    ]
    
    current_context_idx = 0
    step_count = 0
    last_step_time = time.time()
    
    print("Starting visualization (press ESC to exit, SPACE to step)...")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Manual step
                    current_context = test_contexts[current_context_idx % len(test_contexts)]
                    
                    # Generate prediction
                    start_time = time.time()
                    consensus_result = predictor.generate_prediction(
                        current_context, world_graph, step_count
                    )
                    thinking_time = time.time() - start_time
                    
                    # Calculate prediction error (simulated)
                    prediction_error = random.uniform(0.0, 2.0)
                    
                    # Update brain monitor
                    brain_monitor.update(
                        world_graph, 
                        prediction_error, 
                        consensus_result.prediction.motor_action,
                        step_count,
                        consensus_result
                    )
                    
                    # Print prediction info
                    print(f"Step {step_count}: {consensus_result.consensus_strength} consensus")
                    print(f"  Agreement: {consensus_result.agreement_count}/{consensus_result.total_traversals}")
                    print(f"  Action: {consensus_result.prediction.motor_action}")
                    print(f"  Thinking time: {thinking_time:.4f}s")
                    print(f"  Prediction error: {prediction_error:.3f}")
                    
                    step_count += 1
                    current_context_idx += 1
        
        # Auto-step every 2 seconds
        current_time = time.time()
        if current_time - last_step_time >= 2.0:
            current_context = test_contexts[current_context_idx % len(test_contexts)]
            
            # Generate prediction
            start_time = time.time()
            consensus_result = predictor.generate_prediction(
                current_context, world_graph, step_count
            )
            thinking_time = time.time() - start_time
            
            # Calculate prediction error (simulated)
            prediction_error = random.uniform(0.0, 2.0)
            
            # Update brain monitor
            brain_monitor.update(
                world_graph, 
                prediction_error, 
                consensus_result.prediction.motor_action,
                step_count,
                consensus_result
            )
            
            step_count += 1
            current_context_idx += 1
            last_step_time = current_time
        
        # Render
        screen.fill((0, 0, 0))
        brain_monitor.render(screen, 0, 0)
        
        # Show instructions at bottom to avoid overlap
        font = pygame.font.Font(None, 18)
        instructions = [
            "ESC: Exit | SPACE: Manual step | Auto-step every 2s",
            f"Step: {step_count} | Context: {current_context_idx % len(test_contexts)}"
        ]
        
        y_offset = 650  # Near bottom of screen
        for instruction in instructions:
            text = font.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect()
            text_rect.centerx = 250  # Center horizontally
            text_rect.y = y_offset
            screen.blit(text, text_rect)
            y_offset += 20
        
        pygame.display.flip()
        clock.tick(60)
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    predictor_stats = predictor.get_prediction_statistics()
    for key, value in predictor_stats.items():
        print(f"{key}: {value}")
    
    pygame.quit()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()