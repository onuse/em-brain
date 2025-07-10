#!/usr/bin/env python3
"""
Async Brain Observer - Non-blocking GUI that observes brain state.

This replaces the coupled GUI with a proper observer pattern that:
- Receives brain state updates via queue
- Renders at comfortable GUI rate (30 FPS)
- Never blocks brain processing
- Can be disconnected/reconnected without affecting brain
"""

import threading
import time
import queue
from typing import Optional, Dict, Any
import pygame
from core.async_brain_server import AsyncBrainServer, BrainState
from .brain_monitor import BrainStateMonitor
from .grid_world_viz import GridWorldVisualizer


class AsyncBrainObserver:
    """
    Non-blocking brain observer that displays brain state without interference.
    
    This is the proper way to observe any robot brain - physical or simulated.
    """
    
    def __init__(self, brain_server: AsyncBrainServer, brainstem, cell_size: int = 15):
        self.brain_server = brain_server
        self.brainstem = brainstem
        
        # Calculate layout dimensions
        grid_width = brainstem.simulation.width * cell_size
        grid_height = brainstem.simulation.height * cell_size
        brain_panel_width = 400
        
        # Total window size
        self.window_width = grid_width + brain_panel_width
        self.window_height = max(grid_height + 100, 600)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Robot Brain Observer - Decoupled from Brain Processing")
        
        # Create visualization components
        self.grid_viz = GridWorldVisualizer(brainstem, cell_size, ui_width=0)
        self.brain_monitor = BrainStateMonitor(brain_panel_width, self.window_height)
        
        # Override grid visualizer's screen to use our subsection
        self.grid_viz.screen = self.screen.subsurface((0, 0, grid_width, grid_height))
        self.grid_viz.window_width = grid_width
        
        # Control state
        self.running = False
        self.paused = False
        self.gui_fps = 30
        self.clock = pygame.time.Clock()
        
        # Observer state
        self.observer_id = f"gui_observer_{int(time.time())}"
        self.state_queue: Optional[queue.Queue] = None
        self.current_brain_state: Optional[BrainState] = None
        self.last_state_update = time.time()
        
        # GUI performance tracking
        self.gui_frame_count = 0
        self.gui_fps_actual = 0.0
        self.gui_last_fps_update = time.time()
    
    def start_observing(self, update_frequency: float = 30.0):
        """Start observing the brain server."""
        # Register as observer
        self.state_queue = self.brain_server.register_observer(
            self.observer_id,
            update_frequency=update_frequency
        )
        
        self.running = True
        print(f"👁️  GUI Observer started - watching brain at {update_frequency} Hz")
    
    def stop_observing(self):
        """Stop observing the brain server."""
        self.running = False
        
        if self.state_queue:
            self.brain_server.unregister_observer(self.observer_id)
            self.state_queue = None
        
        print("👁️  GUI Observer stopped")
    
    def run(self):
        """Main GUI loop - runs independently of brain processing."""
        self.start_observing()
        
        try:
            while self.running:
                # Handle events
                self._handle_events()
                
                # Update brain state from queue (non-blocking)
                self._update_brain_state()
                
                # Render frame
                self._render_frame()
                
                # Control GUI frame rate
                self.clock.tick(self.gui_fps)
                self._update_gui_fps()
                
        except KeyboardInterrupt:
            print("\n👁️  GUI Observer interrupted")
        finally:
            self.stop_observing()
            pygame.quit()
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"GUI {'paused' if self.paused else 'resumed'} (brain continues running)")
                elif event.key == pygame.K_r:
                    # Reset only affects the visualization, not the brain
                    self.current_brain_state = None
                    print("GUI reset (brain continues running)")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.gui_fps = min(120, self.gui_fps + 5)
                    print(f"GUI FPS: {self.gui_fps}")
                elif event.key == pygame.K_MINUS:
                    self.gui_fps = max(5, self.gui_fps - 5)
                    print(f"GUI FPS: {self.gui_fps}")
                elif event.key == pygame.K_f:
                    # Show brain performance stats
                    stats = self.brain_server.get_performance_stats()
                    print(f"🧠 Brain Performance: {stats['brain_fps']:.1f} FPS, {stats['step_count']} steps")
    
    def _update_brain_state(self):
        """Update brain state from queue (non-blocking)."""
        if not self.state_queue:
            return
        
        # Get all available states (keep only the latest)
        latest_state = None
        states_processed = 0
        
        try:
            while True:
                latest_state = self.state_queue.get_nowait()
                states_processed += 1
        except queue.Empty:
            pass
        
        # Update current state if we got a new one
        if latest_state:
            self.current_brain_state = latest_state
            self.last_state_update = time.time()
            
            # Occasionally print update info
            if states_processed > 1 and latest_state.step_count % 1000 == 0:
                print(f"📊 State update: {latest_state.step_count} steps, "
                      f"{latest_state.node_count} nodes, "
                      f"{latest_state.fps:.1f} brain FPS")
    
    def _render_frame(self):
        """Render the GUI frame."""
        if self.paused:
            return
        
        # Clear screen
        self.screen.fill((20, 20, 20))
        
        # Render grid world
        self.grid_viz.render()
        
        # Render brain monitor with current state
        if self.current_brain_state:
            self._update_brain_monitor()
        
        # Render brain monitor
        brain_x = self.grid_viz.grid_width
        self.brain_monitor.render(self.screen, brain_x, 0)
        
        # Draw separator line
        separator_x = brain_x
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (separator_x, 0), (separator_x, self.window_height), 2)
        
        # Draw status bar
        self._draw_status_bar()
        
        # Update display
        pygame.display.flip()
    
    def _update_brain_monitor(self):
        """Update brain monitor with current brain state."""
        state = self.current_brain_state
        
        # Create a world graph proxy for the monitor
        from core.world_graph import WorldGraph
        graph_proxy = WorldGraph()
        graph_proxy.nodes = {}  # Empty nodes dict
        graph_proxy.total_nodes_created = state.node_count
        
        # Override statistics method to return brain state data
        def get_stats_proxy(approximate=False):
            return {
                'total_nodes': state.node_count,
                'avg_strength': 1.0,  # Placeholder
                'max_strength': 2.0,  # Placeholder  
                'total_merges': 0,    # Placeholder
                'total_accesses': state.step_count,
                'temporal_chain_length': state.step_count,
                'similarity_engine': {'queries_processed': 0}
            }
        
        graph_proxy.get_graph_statistics = get_stats_proxy
        
        # Update monitor
        self.brain_monitor.update(
            graph=graph_proxy,
            prediction_error=state.prediction_error,
            recent_action=state.recent_action,
            step=state.step_count,
            robot_mood=state.drive_states.get('mood', {})
        )
    
    def _draw_status_bar(self):
        """Draw status information at the bottom."""
        status_height = 30
        status_rect = pygame.Rect(0, self.window_height - status_height, 
                                 self.window_width, status_height)
        pygame.draw.rect(self.screen, (40, 40, 40), status_rect)
        
        # Status text
        status_parts = []
        
        if self.paused:
            status_parts.append("GUI PAUSED")
        else:
            status_parts.append("OBSERVING")
        
        # Brain performance
        if self.current_brain_state:
            status_parts.append(f"Brain: {self.current_brain_state.fps:.0f} FPS")
            status_parts.append(f"Nodes: {self.current_brain_state.node_count}")
            status_parts.append(f"Steps: {self.current_brain_state.step_count}")
            
            # Robot state
            status_parts.append(f"Health: {self.current_brain_state.robot_health:.2f}")
            status_parts.append(f"Energy: {self.current_brain_state.robot_energy:.2f}")
        
        # GUI performance
        status_parts.append(f"GUI: {self.gui_fps_actual:.1f} FPS")
        
        # Data freshness
        if self.current_brain_state:
            age = time.time() - self.last_state_update
            if age > 1.0:
                status_parts.append(f"STALE ({age:.1f}s)")
            else:
                status_parts.append("LIVE")
        
        status_text = " | ".join(status_parts)
        
        font = pygame.font.Font(None, 16)
        text_surface = font.render(status_text, True, (255, 255, 255))
        
        # Center the text
        text_rect = text_surface.get_rect()
        text_rect.center = (self.window_width // 2, self.window_height - status_height // 2)
        self.screen.blit(text_surface, text_rect)
        
        # Help text
        help_text = "ESC: Quit | SPACE: Pause GUI | R: Reset GUI | F: Brain Stats | +/-: GUI Speed"
        help_surface = pygame.font.Font(None, 12).render(help_text, True, (150, 150, 150))
        help_rect = help_surface.get_rect()
        help_rect.right = self.window_width - 10
        help_rect.bottom = self.window_height - 5
        self.screen.blit(help_surface, help_rect)
    
    def _update_gui_fps(self):
        """Update GUI FPS counter."""
        self.gui_frame_count += 1
        current_time = time.time()
        
        if current_time - self.gui_last_fps_update >= 1.0:
            self.gui_fps_actual = self.gui_frame_count / (current_time - self.gui_last_fps_update)
            self.gui_frame_count = 0
            self.gui_last_fps_update = current_time


def main():
    """Example usage of the async brain observer."""
    from simulation.brainstem_sim import GridWorldBrainstem
    from core.async_brain_server import AsyncBrainServer
    
    print("🚀 Starting Decoupled Brain System")
    print("=" * 50)
    
    # Initialize brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    brain_server = AsyncBrainServer(brainstem, brainstem.brain_client)
    
    # Start brain server
    brain_server.start()
    
    # Create and run observer
    observer = AsyncBrainObserver(brain_server, brainstem)
    
    try:
        observer.run()
    finally:
        brain_server.stop()


if __name__ == "__main__":
    main()