#!/usr/bin/env python3
"""
Quick test of brain monitor rendering
"""

import pygame
import sys
from visualization.brain_monitor import BrainStateMonitor
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode

def test_brain_monitor():
    """Test brain monitor rendering with dummy data."""
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Brain Monitor Test")
    clock = pygame.time.Clock()
    
    # Create brain monitor
    monitor = BrainStateMonitor(400, 600)
    
    # Create dummy world graph with some data
    world_graph = WorldGraph()
    
    # Add some dummy experiences
    for i in range(5):
        experience = ExperienceNode(
            mental_context=[1.0, 2.0, 3.0],
            action_taken={'forward_motor': 0.5, 'turn_motor': 0.0, 'brake_motor': 0.0},
            predicted_sensory=[0.5, 0.5, 0.5],
            actual_sensory=[0.6, 0.4, 0.5],
            prediction_error=0.1
        )
        world_graph.add_node(experience)
    
    # Update monitor with data
    monitor.update(world_graph, 0.15, {'forward_motor': 0.3}, 10)
    
    print("Brain monitor test started...")
    print("Should see brain monitor panel on the right")
    print("Press ESC or close window to exit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw test grid on left (400px)
        for x in range(0, 400, 50):
            pygame.draw.line(screen, (100, 100, 100), (x, 0), (x, 600))
        for y in range(0, 600, 50):
            pygame.draw.line(screen, (100, 100, 100), (0, y), (400, y))
        
        # Render brain monitor on right (starting at x=400)
        monitor.render(screen, 400, 0)
        
        # Draw separator line
        pygame.draw.line(screen, (255, 255, 255), (400, 0), (400, 600), 2)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("Brain monitor test completed")

if __name__ == "__main__":
    test_brain_monitor()