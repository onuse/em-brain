#!/usr/bin/env python3
"""
Panel Test Demo - Draws a simple colored rectangle where the brain panel should be
"""

import pygame
from core import WorldGraph
from simulation import GridWorldBrainstem
from visualization import IntegratedDisplay


def simple_learning_agent(state_dict) -> dict:
    """Simple reactive learning agent."""
    sensors = state_dict['sensors']
    distance_sensors = sensors[:4]
    front_distance = distance_sensors[0]
    left_distance = distance_sensors[1] 
    right_distance = distance_sensors[2]
    
    if front_distance < 0.3:
        if left_distance > right_distance:
            return {'forward_motor': 0.0, 'turn_motor': -0.7, 'brake_motor': 0.3}
        else:
            return {'forward_motor': 0.0, 'turn_motor': 0.7, 'brake_motor': 0.3}
    else:
        import random
        turn = random.uniform(-0.2, 0.2) if random.random() < 0.1 else 0.0
        return {'forward_motor': 0.4, 'turn_motor': turn, 'brake_motor': 0.0}


class TestIntegratedDisplay(IntegratedDisplay):
    """Test version that draws a bright colored rectangle where brain panel should be."""
    
    def _render_frame(self):
        """Override to add test rectangle."""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Render grid world (left side)
        self.grid_viz.render()
        
        # Draw TEST RECTANGLE where brain panel should be (right side)
        brain_x = self.grid_viz.grid_width
        brain_rect = pygame.Rect(brain_x, 0, 400, self.window_height)
        pygame.draw.rect(self.screen, (255, 0, 255), brain_rect)  # Bright magenta
        
        # Draw text to confirm this is the brain panel area
        font = pygame.font.Font(None, 36)
        text = font.render("BRAIN PANEL", True, (255, 255, 255))
        text_rect = text.get_rect(center=(brain_x + 200, 100))
        self.screen.blit(text, text_rect)
        
        # Draw separator line
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (brain_x, 0), (brain_x, self.window_height), 3)
        
        # Draw status bar at bottom
        self._draw_status_bar()
        
        # Update display
        pygame.display.flip()


def main():
    """Test if the brain panel area is visible."""
    print("🧪 Panel Test Demo")
    print("=" * 40)
    print("Drawing BRIGHT MAGENTA rectangle where brain panel should be")
    print("If you see it, the panel area is working!")
    print("=" * 40)
    
    try:
        brainstem = GridWorldBrainstem(world_width=12, world_height=12, seed=42)
        brain_graph = WorldGraph()
        
        # Use test display that draws colored rectangle
        display = TestIntegratedDisplay(brainstem, cell_size=25)
        display.set_brain_graph(brain_graph)
        display.set_learning_callback(simple_learning_agent)
        
        print(f"🖥️  Window: {display.window_width}x{display.window_height}")
        print("🚀 Look for BRIGHT MAGENTA panel on the right!")
        
        display.run(auto_step=True, step_delay=0.3)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Panel test completed")


if __name__ == "__main__":
    main()