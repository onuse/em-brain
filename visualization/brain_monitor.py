"""
Brain state monitoring and visualization.
Displays real-time information about the robot's learning and decision-making process.
"""

import warnings
# Suppress pygame's pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import pygame
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from datetime import datetime
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from predictor.consensus_resolver import ConsensusResult


class BrainStateMonitor:
    """Monitors and displays brain state information in real-time."""
    
    # Color scheme
    COLORS = {
        'background': (20, 20, 20),
        'panel': (40, 40, 40),
        'text': (255, 255, 255),
        'accent': (0, 150, 255),
        'success': (0, 255, 0),
        'warning': (255, 255, 0),
        'error': (255, 0, 0),
        'graph_line': (0, 200, 255),
        'graph_grid': (80, 80, 80),
        'node_strong': (255, 100, 100),
        'node_medium': (255, 255, 100),
        'node_weak': (100, 255, 100),
    }
    
    def __init__(self, width: int = 400, height: int = 600):
        """Initialize the brain state monitor."""
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        
        # Fonts
        self.font_title = pygame.font.Font(None, 24)
        self.font_normal = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
        
        # Data tracking
        self.graph_stats_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=50)
        self.learning_events = deque(maxlen=20)
        self.consensus_history = deque(maxlen=50)
        
        # Current state
        self.current_graph: Optional[WorldGraph] = None
        self.current_prediction_error = 0.0
        self.current_node_count = 0
        self.current_step = 0
        self.current_consensus: Optional[ConsensusResult] = None
        
        # Display settings
        self.show_node_graph = True
        self.show_stats_graph = True
        self.auto_scroll_log = True
        
        # Performance optimization
        self.update_counter = 0
        self.stats_update_frequency = 10  # Update expensive stats every 10 frames instead of every frame
    
    def update(self, graph: WorldGraph, prediction_error: float = 0.0, 
               recent_action: Dict[str, float] = None, step: int = 0,
               consensus_result: ConsensusResult = None, brain_client=None,
               robot_mood: Dict = None):
        """Update the brain monitor with new data."""
        self.current_graph = graph
        self.current_prediction_error = prediction_error
        self.current_step = step
        self.current_consensus = consensus_result
        self.brain_client = brain_client
        self.current_robot_mood = robot_mood
        
        # Increment update counter for performance optimization
        self.update_counter += 1
        
        # Debug: Print update info occasionally (reduced frequency)
        if step % 200 == 0:  # Every 200 steps instead of 20
            node_count = graph.node_count() if graph else 0
            print(f"Brain monitor update: step {step}, {node_count} nodes, error {prediction_error:.3f}")
        
        # Update expensive statistics only every N frames for performance
        if self.update_counter % self.stats_update_frequency == 0:
            stats = graph.get_graph_statistics()
            stats['step'] = step
            stats['prediction_error'] = prediction_error
            self.graph_stats_history.append(stats)
        
        # Always update prediction history (lightweight)
        self.prediction_history.append(prediction_error)
        
        # Update consensus history
        if consensus_result:
            self.consensus_history.append({
                'strength': consensus_result.consensus_strength,
                'agreement': consensus_result.agreement_count,
                'total': consensus_result.total_traversals,
                'step': step
            })
        
        # Log significant events (less frequently for performance)
        if recent_action and self.update_counter % 5 == 0:  # Every 5 frames
            self._log_event(f"Action: {self._format_action(recent_action)}", "info")
        
        if consensus_result and self.update_counter % 3 == 0:  # Every 3 frames
            self._log_event(f"Thinking: {consensus_result.consensus_strength} consensus ({consensus_result.agreement_count}/{consensus_result.total_traversals})", "info")
        
        if prediction_error > 2.0:
            self._log_event(f"High prediction error: {prediction_error:.3f}", "warning")
        
        # Only check for merges when we updated stats
        if self.update_counter % self.stats_update_frequency == 0:
            latest_stats = list(self.graph_stats_history)
            if len(latest_stats) > 1:
                prev_merges = latest_stats[-2].get('total_merges', 0)
                current_merges = latest_stats[-1].get('total_merges', 0)
                if current_merges > prev_merges:
                    new_merges = current_merges - prev_merges
                    self._log_event(f"Memory consolidation: {new_merges} nodes merged", "success")
    
    def render(self, target_surface: pygame.Surface, x: int, y: int):
        """Render the brain monitor to the target surface."""
        try:
            # Clear background
            self.surface.fill(self.COLORS['background'])
            
            # Draw sections
            y_offset = 10
            y_offset = self._draw_title("Brain State", y_offset)
            y_offset = self._draw_current_stats(y_offset)
            y_offset = self._draw_robot_mood(y_offset)
            y_offset = self._draw_graph_visualization(y_offset)
            y_offset = self._draw_stats_graph(y_offset)
            y_offset = self._draw_event_log(y_offset)
            
            # Blit to target surface
            target_surface.blit(self.surface, (x, y))
            
        except Exception as e:
            # If rendering fails, at least show an error message
            print(f"Brain monitor render error: {e}")
            error_font = pygame.font.Font(None, 24)
            error_text = error_font.render(f"Render Error: {str(e)[:30]}", True, (255, 0, 0))
            target_surface.blit(error_text, (x + 10, y + 10))
    
    def _draw_title(self, title: str, y: int) -> int:
        """Draw section title."""
        title_text = self.font_title.render(title, True, self.COLORS['accent'])
        title_rect = title_text.get_rect()
        title_rect.centerx = self.width // 2
        title_rect.y = y
        
        self.surface.blit(title_text, title_rect)
        
        # Underline
        line_y = y + title_rect.height + 2
        pygame.draw.line(self.surface, self.COLORS['accent'], 
                        (10, line_y), (self.width - 10, line_y), 2)
        
        return line_y + 10
    
    def _draw_current_stats(self, y: int) -> int:
        """Draw current brain statistics."""
        if not self.current_graph:
            no_data_text = self.font_normal.render("No brain data", True, self.COLORS['text'])
            self.surface.blit(no_data_text, (10, y))
            # Draw a test rectangle to verify this section is being called
            pygame.draw.rect(self.surface, (50, 50, 150), (10, y + 30, 200, 20))
            test_text = self.font_small.render("Brain monitor active", True, self.COLORS['text'])
            self.surface.blit(test_text, (15, y + 33))
            return y + 60
        
        # Use cached stats if available, otherwise get fresh stats
        if hasattr(self, '_cached_display_stats') and self.update_counter % self.stats_update_frequency != 0:
            stats = self._cached_display_stats
        else:
            stats = self.current_graph.get_graph_statistics()
            self._cached_display_stats = stats
        
        # Stats to display
        stat_items = [
            ("Nodes", f"{stats['total_nodes']}"),
            ("Avg Strength", f"{stats['avg_strength']:.2f}"),
            ("Max Strength", f"{stats['max_strength']:.2f}"),
            ("Total Merges", f"{stats['total_merges']}"),
            ("Total Accesses", f"{stats['total_accesses']}"),
            ("Prediction Error", f"{self.current_prediction_error:.3f}"),
        ]
        
        # Add consensus info if available
        if self.current_consensus:
            stat_items.append(("Consensus", f"{self.current_consensus.consensus_strength}"))
            stat_items.append(("Agreement", f"{self.current_consensus.agreement_count}/{self.current_consensus.total_traversals}"))
        
        for label, value in stat_items:
            # Label
            label_text = self.font_normal.render(f"{label}:", True, self.COLORS['text'])
            self.surface.blit(label_text, (10, y))
            
            # Value with color coding
            color = self.COLORS['text']
            if label == "Prediction Error":
                if self.current_prediction_error > 2.0:
                    color = self.COLORS['error']
                elif self.current_prediction_error > 1.0:
                    color = self.COLORS['warning']
                else:
                    color = self.COLORS['success']
            elif label == "Total Merges" and stats['total_merges'] > 0:
                color = self.COLORS['success']
            elif label == "Consensus" and self.current_consensus:
                if self.current_consensus.consensus_strength == "perfect":
                    color = self.COLORS['success']
                elif self.current_consensus.consensus_strength == "strong":
                    color = self.COLORS['success']
                elif self.current_consensus.consensus_strength == "weak":
                    color = self.COLORS['warning']
                else:
                    color = self.COLORS['error']
            
            value_text = self.font_normal.render(value, True, color)
            value_rect = value_text.get_rect()
            value_rect.right = self.width - 10
            value_rect.y = y
            self.surface.blit(value_text, value_rect)
            
            y += 22
        
        return y + 10
    
    def _draw_robot_mood(self, y: int) -> int:
        """Draw the robot's current emotional state (your aesthetic idea!)."""
        # Section title
        mood_title = self.font_title.render("Robot Mood", True, self.COLORS['accent'])
        self.surface.blit(mood_title, (10, y))
        y += 25
        
        # Get actual mood from brain system
        mood_descriptor = "unknown"
        satisfaction = 0.0  # Default to neutral
        urgency = 0.0
        
        # Access mood if available
        if hasattr(self, 'current_robot_mood') and self.current_robot_mood:
            mood_data = self.current_robot_mood
            mood_descriptor = mood_data.get('mood_descriptor', 'unknown')
            satisfaction = mood_data.get('overall_satisfaction', 0.0)
            urgency = mood_data.get('overall_urgency', 0.0)
        else:
            # Try to get mood from brain if available
            if hasattr(self, 'brain_client') and self.brain_client:
                try:
                    # Get motivation system and calculate mood
                    motivation_stats = self.brain_client.get_motivation_statistics()
                    if motivation_stats and 'mood' in motivation_stats:
                        mood_data = motivation_stats['mood']
                        mood_descriptor = mood_data.get('mood_descriptor', 'unknown')
                        satisfaction = mood_data.get('overall_satisfaction', 0.0)
                        urgency = mood_data.get('overall_urgency', 0.0)
                except Exception as e:
                    pass  # Use defaults if mood system unavailable
        
        # Display mood descriptor
        mood_color = self.COLORS['success'] if satisfaction > 0.5 else self.COLORS['warning'] if satisfaction > 0.0 else self.COLORS['error']
        mood_text = self.font_normal.render(f"Mood: {mood_descriptor}", True, mood_color)
        self.surface.blit(mood_text, (10, y))
        
        # Draw dynamic mood bars
        bar_y = y + 20
        bar_width = 150
        bar_height = 6
        
        # Satisfaction bar (green = good, red = bad)
        sat_label = self.font_small.render("Satisfaction:", True, self.COLORS['text'])
        self.surface.blit(sat_label, (10, bar_y))
        
        sat_rect = pygame.Rect(100, bar_y + 2, bar_width, bar_height)
        pygame.draw.rect(self.surface, self.COLORS['graph_grid'], sat_rect)
        
        # Dynamic satisfaction level
        sat_normalized = max(0.0, min(1.0, (satisfaction + 1.0) / 2.0))  # Convert -1,+1 to 0,1
        sat_fill_width = int(bar_width * sat_normalized)
        sat_color = self.COLORS['success'] if satisfaction > 0 else self.COLORS['error']
        if sat_fill_width > 0:
            sat_fill = pygame.Rect(100, bar_y + 2, sat_fill_width, bar_height)
            pygame.draw.rect(self.surface, sat_color, sat_fill)
        
        # Urgency bar (red = urgent, green = calm)
        urgency_y = bar_y + 15
        urgency_label = self.font_small.render("Urgency:", True, self.COLORS['text'])
        self.surface.blit(urgency_label, (10, urgency_y))
        
        urgency_rect = pygame.Rect(100, urgency_y + 2, bar_width, bar_height)
        pygame.draw.rect(self.surface, self.COLORS['graph_grid'], urgency_rect)
        
        # Dynamic urgency level
        urgency_normalized = max(0.0, min(1.0, urgency))
        urgency_fill_width = int(bar_width * urgency_normalized)
        urgency_color = self.COLORS['error'] if urgency > 0.7 else self.COLORS['warning'] if urgency > 0.3 else self.COLORS['success']
        if urgency_fill_width > 0:
            urgency_fill = pygame.Rect(100, urgency_y + 2, urgency_fill_width, bar_height)
            pygame.draw.rect(self.surface, urgency_color, urgency_fill)
        
        return y + 50
    
    def _draw_graph_visualization(self, y: int) -> int:
        """Draw a simplified visualization of the memory graph."""
        if not self.current_graph or not self.show_node_graph:
            return y
        
        # Section header
        header_text = self.font_normal.render("Memory Graph", True, self.COLORS['accent'])
        self.surface.blit(header_text, (10, y))
        y += 25
        
        # Graph area
        graph_rect = pygame.Rect(10, y, self.width - 20, 120)
        pygame.draw.rect(self.surface, self.COLORS['panel'], graph_rect)
        pygame.draw.rect(self.surface, self.COLORS['accent'], graph_rect, 1)
        
        # Get strongest nodes for visualization
        strongest_nodes = self.current_graph.get_strongest_nodes(20)
        
        if strongest_nodes:
            # Draw nodes as circles with size based on strength
            import math
            nodes_per_row = 8
            node_spacing = graph_rect.width // (nodes_per_row + 1)
            row_spacing = graph_rect.height // 4
            
            for i, node in enumerate(strongest_nodes[:16]):  # Max 16 nodes (2 rows)
                row = i // nodes_per_row
                col = i % nodes_per_row
                
                center_x = graph_rect.left + (col + 1) * node_spacing
                center_y = graph_rect.top + (row + 1) * row_spacing + 20
                
                # Node size based on strength (3-12 pixels radius)
                radius = max(3, min(12, int(node.strength * 6)))
                
                # Color based on strength
                if node.strength > 2.0:
                    color = self.COLORS['node_strong']
                elif node.strength > 1.5:
                    color = self.COLORS['node_medium']
                else:
                    color = self.COLORS['node_weak']
                
                # Draw node
                pygame.draw.circle(self.surface, color, (center_x, center_y), radius)
                
                # Draw access indicator
                if node.times_accessed > 0:
                    pygame.draw.circle(self.surface, self.COLORS['text'], 
                                     (center_x, center_y), radius, 1)
        
        # Graph stats text
        stats = self.current_graph.get_graph_statistics()
        graph_info = f"Showing {min(len(strongest_nodes), 16)} strongest of {stats['total_nodes']} nodes"
        info_text = self.font_small.render(graph_info, True, self.COLORS['text'])
        self.surface.blit(info_text, (graph_rect.left + 5, graph_rect.bottom - 15))
        
        return graph_rect.bottom + 15
    
    def _draw_stats_graph(self, y: int) -> int:
        """Draw a line graph of statistics over time."""
        if not self.graph_stats_history or not self.show_stats_graph:
            return y
        
        # Section header
        header_text = self.font_normal.render("Learning Progress", True, self.COLORS['accent'])
        self.surface.blit(header_text, (10, y))
        y += 25
        
        # Graph area
        graph_rect = pygame.Rect(10, y, self.width - 20, 80)
        pygame.draw.rect(self.surface, self.COLORS['panel'], graph_rect)
        pygame.draw.rect(self.surface, self.COLORS['accent'], graph_rect, 1)
        
        # Draw grid lines
        for i in range(1, 4):
            grid_y = graph_rect.top + i * (graph_rect.height // 4)
            pygame.draw.line(self.surface, self.COLORS['graph_grid'],
                           (graph_rect.left, grid_y), (graph_rect.right, grid_y))
        
        # Draw prediction error line
        if len(self.prediction_history) > 1:
            points = []
            max_error = max(self.prediction_history) if self.prediction_history else 1.0
            max_error = max(max_error, 0.1)  # Avoid division by zero
            
            for i, error in enumerate(self.prediction_history):
                x = graph_rect.left + (i * graph_rect.width) // len(self.prediction_history)
                y_val = graph_rect.bottom - (error / max_error) * graph_rect.height
                y_val = max(graph_rect.top, min(graph_rect.bottom, y_val))
                points.append((x, y_val))
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, self.COLORS['graph_line'], False, points, 2)
        
        # Graph label
        label_text = self.font_small.render("Prediction Error", True, self.COLORS['text'])
        self.surface.blit(label_text, (graph_rect.left + 5, graph_rect.top + 5))
        
        return graph_rect.bottom + 15
    
    def _draw_event_log(self, y: int) -> int:
        """Draw the recent learning events log."""
        # Section header
        header_text = self.font_normal.render("Learning Events", True, self.COLORS['accent'])
        self.surface.blit(header_text, (10, y))
        y += 25
        
        # Log area
        log_rect = pygame.Rect(10, y, self.width - 20, self.height - y - 10)
        pygame.draw.rect(self.surface, self.COLORS['panel'], log_rect)
        pygame.draw.rect(self.surface, self.COLORS['accent'], log_rect, 1)
        
        # Draw events
        event_y = log_rect.top + 5
        line_height = 16
        
        # Show recent events (newest first)
        visible_events = list(self.learning_events)[-15:]  # Last 15 events
        visible_events.reverse()  # Newest first
        
        for event in visible_events:
            if event_y + line_height > log_rect.bottom - 5:
                break
            
            # Event color based on type
            color = self.COLORS['text']
            if event['type'] == 'success':
                color = self.COLORS['success']
            elif event['type'] == 'warning':
                color = self.COLORS['warning']
            elif event['type'] == 'error':
                color = self.COLORS['error']
            
            # Event text
            event_text = self.font_small.render(event['message'], True, color)
            
            # Truncate if too long
            if event_text.get_width() > log_rect.width - 10:
                truncated_msg = event['message'][:50] + "..."
                event_text = self.font_small.render(truncated_msg, True, color)
            
            self.surface.blit(event_text, (log_rect.left + 5, event_y))
            event_y += line_height
        
        return self.height
    
    def _log_event(self, message: str, event_type: str = "info"):
        """Add an event to the log."""
        event = {
            'message': message,
            'type': event_type,
            'timestamp': datetime.now(),
            'step': self.current_step
        }
        self.learning_events.append(event)
    
    def _format_action(self, action: Dict[str, float]) -> str:
        """Format action dictionary for display."""
        if not action:
            return "None"
        
        formatted_parts = []
        for key, value in action.items():
            if abs(value) > 0.1:  # Only show significant actions
                formatted_parts.append(f"{key[:4]}:{value:.2f}")
        
        return " ".join(formatted_parts) if formatted_parts else "idle"
    
    def toggle_node_graph(self):
        """Toggle node graph display."""
        self.show_node_graph = not self.show_node_graph
    
    def toggle_stats_graph(self):
        """Toggle statistics graph display."""
        self.show_stats_graph = not self.show_stats_graph
    
    def clear_log(self):
        """Clear the event log."""
        self.learning_events.clear()