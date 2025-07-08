"""
Integrated display system combining grid world visualization and brain monitoring.
Provides a comprehensive real-time view of the robot's world and mind.
"""

import pygame
import time
from typing import Dict, Any, Optional, Callable
from simulation.brainstem_sim import GridWorldBrainstem
from core.world_graph import WorldGraph
from .grid_world_viz import GridWorldVisualizer
from .brain_monitor import BrainStateMonitor


class IntegratedDisplay:
    """
    Complete visualization system showing both the physical world and mental state.
    Implements the multi-panel interface: Grid World | Brain State
                                         Memory Graph | Live Log
    """
    
    def __init__(self, brainstem: GridWorldBrainstem, cell_size: int = 25):
        """
        Initialize the integrated display system.
        
        Args:
            brainstem: The brainstem simulation to visualize
            cell_size: Size of grid cells in pixels
        """
        self.brainstem = brainstem
        self.brain_graph: Optional[WorldGraph] = None
        
        # Calculate layout dimensions
        grid_width = brainstem.simulation.width * cell_size
        grid_height = brainstem.simulation.height * cell_size
        brain_panel_width = 400
        
        # Total window size
        self.window_width = grid_width + brain_panel_width
        self.window_height = max(grid_height, 600)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Emergent Intelligence Robot - Complete System Monitor")
        
        # Create visualizer components
        self.grid_viz = GridWorldVisualizer(brainstem, cell_size, ui_width=0)  # No UI, we handle it
        self.brain_monitor = BrainStateMonitor(brain_panel_width, self.window_height)
        
        # Override grid visualizer's screen to use our subsection
        self.grid_viz.screen = self.screen.subsurface((0, 0, grid_width, grid_height))
        self.grid_viz.window_width = grid_width
        
        # Control state
        self.running = True
        self.paused = False
        self.step_mode = False
        self.fps = 30
        self.clock = pygame.time.Clock()
        
        # Learning integration
        self.learning_callback: Optional[Callable] = None
        self.action_generator: Optional[Callable] = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
    
    def set_brain_graph(self, graph: WorldGraph):
        """Set the brain graph to monitor."""
        self.brain_graph = graph
    
    def set_learning_callback(self, callback: Callable[[Dict[str, Any]], Dict[str, float]]):
        """
        Set a callback function for learning integration.
        
        The callback should take current state and return motor actions.
        Function signature: (state_dict) -> motor_commands_dict
        """
        self.learning_callback = callback
    
    def set_action_generator(self, generator: Callable[[], Dict[str, float]]):
        """
        Set a simple action generator function.
        
        Function signature: () -> motor_commands_dict
        """
        self.action_generator = generator
    
    def run(self, auto_step: bool = True, step_delay: float = 0.1):
        """
        Run the integrated visualization with automatic stepping.
        
        Args:
            auto_step: Whether to automatically step the simulation
            step_delay: Delay between automatic steps (seconds)
        """
        last_auto_step = time.time()
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Automatic stepping
            if auto_step and not self.paused and not self.step_mode:
                current_time = time.time()
                if current_time - last_auto_step >= step_delay:
                    self._perform_simulation_step()
                    last_auto_step = current_time
            
            # Render frame
            self._render_frame()
            
            # Control frame rate
            self.clock.tick(self.fps)
            self._update_fps_counter()
        
        self.cleanup()
    
    def step_once(self):
        """Perform a single simulation step (useful for manual stepping)."""
        self._perform_simulation_step()
        self._render_frame()
        pygame.display.flip()
    
    def _handle_events(self):
        """Handle all pygame events and user input."""
        events = self.grid_viz.handle_events()
        
        # Process control events
        if events['quit']:
            self.running = False
        
        if events['pause']:
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
        
        if events['reset']:
            self.brainstem.reset_robot()
            if self.brain_graph:
                # Reset graph or create new one
                self.brain_graph = WorldGraph()
            self.brain_monitor.clear_log()
            print("System reset")
        
        if events['step']:
            if self.paused:
                self._perform_simulation_step()
                print("Manual step performed")
        
        # Additional keyboard handling
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    self._toggle_help()
                elif event.key == pygame.K_n:
                    self.brain_monitor.toggle_node_graph()
                elif event.key == pygame.K_l:
                    self.brain_monitor.toggle_stats_graph()
                elif event.key == pygame.K_c:
                    self.brain_monitor.clear_log()
                elif event.key == pygame.K_t:
                    self.step_mode = not self.step_mode
                    print(f"Step mode {'enabled' if self.step_mode else 'disabled'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(120, self.fps + 5)
                    print(f"FPS: {self.fps}")
                elif event.key == pygame.K_MINUS:
                    self.fps = max(5, self.fps - 5)
                    print(f"FPS: {self.fps}")
    
    def _perform_simulation_step(self):
        """Perform one step of the simulation with learning integration."""
        # Get current state
        sensor_packet = self.brainstem.get_sensor_readings()
        
        # Generate action
        motor_commands = self._generate_motor_action(sensor_packet)
        
        # Execute action
        is_alive = self.brainstem.execute_motor_commands(motor_commands)
        
        # Get new state
        new_sensor_packet = self.brainstem.get_sensor_readings()
        
        # Calculate prediction error (simplified for demo)
        prediction_error = self._calculate_prediction_error(sensor_packet, new_sensor_packet)
        
        # Update brain monitor
        if self.brain_graph:
            step_count = self.brainstem.get_simulation_stats()['step_count']
            
            # Get mood data from brain system
            robot_mood = None
            try:
                # Access motivation system through predictor
                predictor = self.brainstem.brain_client.get_predictor()
                if hasattr(predictor, 'motivation_system'):
                    motivation_stats = predictor.motivation_system.get_motivation_statistics()
                    if 'mood' in motivation_stats:
                        robot_mood = motivation_stats['mood']
                        # Debug print occasionally to verify mood is changing
                        if step_count % 500 == 0:
                            print(f"🧠 Robot mood: {robot_mood.get('mood_descriptor', 'unknown')} (satisfaction: {robot_mood.get('overall_satisfaction', 0.0):.2f}, urgency: {robot_mood.get('overall_urgency', 0.0):.2f})")
            except Exception as e:
                # Debug: print error occasionally to diagnose issues
                if step_count % 1000 == 0:
                    print(f"Warning: Could not get robot mood: {e}")
                pass  # Use defaults if mood system unavailable
            
            self.brain_monitor.update(
                self.brain_graph,
                prediction_error,
                motor_commands,
                step_count,
                brain_client=self.brainstem.brain_client,
                robot_mood=robot_mood
            )
            # Debug print occasionally (reduced frequency)
            if step_count % 500 == 0:
                print(f"Integrated display: updating brain monitor at step {step_count}")
        else:
            # Debug: Still update monitor even without graph to test rendering
            step_count = self.brainstem.get_simulation_stats()['step_count']
            
            # Get mood data even without brain graph
            robot_mood = None
            try:
                predictor = self.brainstem.brain_client.get_predictor()
                if hasattr(predictor, 'motivation_system'):
                    motivation_stats = predictor.motivation_system.get_motivation_statistics()
                    if 'mood' in motivation_stats:
                        robot_mood = motivation_stats['mood']
            except Exception:
                pass
            
            self.brain_monitor.update(None, prediction_error, motor_commands, step_count, 
                                    brain_client=self.brainstem.brain_client, robot_mood=robot_mood)
            if step_count % 500 == 0:
                print(f"Integrated display: no brain graph available at step {step_count}")
            
            # Note: Experience creation is handled by the brain interface, not here
        
        # Handle robot death
        if not is_alive:
            self.brain_monitor._log_event("Robot died - resetting", "error")
            # No sleep delay - let the brain continue learning immediately
    
    def _generate_motor_action(self, sensor_packet) -> Dict[str, float]:
        """Generate motor action using available methods."""
        # Priority: learning callback > action generator > random
        if self.learning_callback:
            state = {
                'sensors': sensor_packet.sensor_values,
                'position': self.brainstem.simulation.robot.position,
                'orientation': self.brainstem.simulation.robot.orientation,
                'health': self.brainstem.simulation.robot.health,
                'energy': self.brainstem.simulation.robot.energy
            }
            return self.learning_callback(state)
        
        elif self.action_generator:
            return self.action_generator()
        
        else:
            # Simple random exploration
            import random
            return {
                'forward_motor': random.uniform(-0.3, 0.7),  # Bias toward forward
                'turn_motor': random.uniform(-0.5, 0.5),
                'brake_motor': random.uniform(0.0, 0.2)
            }
    
    def _calculate_prediction_error(self, old_packet, new_packet) -> float:
        """Calculate prediction error between sensor readings."""
        # Simplified: assume we predicted no change
        old_sensors = old_packet.sensor_values
        new_sensors = new_packet.sensor_values
        
        if len(old_sensors) != len(new_sensors):
            return 10.0  # High error for mismatched lengths
        
        # RMS error
        error_sum = sum((new - old) ** 2 for old, new in zip(old_sensors, new_sensors))
        return (error_sum / len(old_sensors)) ** 0.5
    
    
    def _render_frame(self):
        """Render a complete frame of the visualization."""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Render grid world (left side)
        self.grid_viz.render()
        
        # Render brain monitor (right side)
        brain_x = self.grid_viz.grid_width
        self.brain_monitor.render(self.screen, brain_x, 0)
        
        # Draw separator line
        separator_x = brain_x
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (separator_x, 0), (separator_x, self.window_height), 2)
        
        # Draw status bar at bottom
        self._draw_status_bar()
        
        # Update display
        pygame.display.flip()
    
    def _draw_status_bar(self):
        """Draw status information at the bottom of the screen."""
        status_height = 25
        status_rect = pygame.Rect(0, self.window_height - status_height, self.window_width, status_height)
        pygame.draw.rect(self.screen, (40, 40, 40), status_rect)
        
        # Status text
        status_parts = []
        
        if self.paused:
            status_parts.append("PAUSED")
        elif self.step_mode:
            status_parts.append("STEP MODE")
        else:
            status_parts.append("RUNNING")
        
        status_parts.append(f"FPS: {self.current_fps:.1f}")
        
        if self.brain_graph:
            status_parts.append(f"Nodes: {self.brain_graph.node_count()}")
        
        sim_stats = self.brainstem.get_simulation_stats()
        status_parts.append(f"Step: {sim_stats['step_count']}")
        status_parts.append(f"Health: {sim_stats['robot_health']:.2f}")
        
        status_text = " | ".join(status_parts)
        
        font = pygame.font.Font(None, 18)
        text_surface = font.render(status_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, self.window_height - 20))
        
        # Help hint
        help_text = "F1: Help | SPACE: Pause | T: Step Mode | +/-: Speed"
        help_surface = font.render(help_text, True, (150, 150, 150))
        help_rect = help_surface.get_rect()
        help_rect.right = self.window_width - 10
        help_rect.y = self.window_height - 20
        self.screen.blit(help_surface, help_rect)
    
    def _update_fps_counter(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def _toggle_help(self):
        """Display help information."""
        help_text = [
            "EMERGENT INTELLIGENCE ROBOT - CONTROLS",
            "",
            "Simulation Control:",
            "  SPACE - Pause/Resume simulation",
            "  T - Toggle step mode (manual stepping)",
            "  ENTER - Single step (when paused/step mode)",
            "  R - Reset robot and world",
            "  ESC - Exit application",
            "",
            "Display Control:",
            "  S - Toggle sensor rays",
            "  V - Toggle vision overlay", 
            "  G - Toggle grid lines",
            "  N - Toggle node graph display",
            "  L - Toggle learning graph",
            "  C - Clear event log",
            "",
            "Speed Control:",
            "  + - Increase FPS",
            "  - - Decrease FPS",
            "",
            "F1 - Show this help",
            "",
            "Press any key to continue..."
        ]
        
        # Create help overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(200)
        self.screen.blit(overlay, (0, 0))
        
        # Draw help text
        font = pygame.font.Font(None, 24)
        y = 50
        
        for line in help_text:
            if line.startswith("EMERGENT"):
                color = (0, 150, 255)
                text_font = pygame.font.Font(None, 28)
            elif line.endswith(":"):
                color = (255, 255, 0)
                text_font = font
            else:
                color = (255, 255, 255)
                text_font = font
            
            text_surface = text_font.render(line, True, color)
            text_rect = text_surface.get_rect()
            text_rect.centerx = self.window_width // 2
            text_rect.y = y
            self.screen.blit(text_surface, text_rect)
            y += 25
        
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting = False
    
    def cleanup(self):
        """Clean up resources."""
        self.grid_viz.cleanup()
        pygame.quit()
        
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state for external analysis."""
        return {
            'brainstem_stats': self.brainstem.get_simulation_stats(),
            'world_state': self.brainstem.get_world_state(),
            'brain_nodes': self.brain_graph.node_count() if self.brain_graph else 0,
            'brain_stats': self.brain_graph.get_graph_statistics() if self.brain_graph else {},
            'running': self.running,
            'paused': self.paused,
            'fps': self.current_fps
        }