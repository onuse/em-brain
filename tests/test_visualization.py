"""
Unit tests for visualization components.
Tests the visualization system without requiring a display.
"""

import pytest
import pygame
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.grid_world_viz import GridWorldVisualizer
from visualization.brain_monitor import BrainStateMonitor
from visualization.integrated_display import IntegratedDisplay


class TestVisualizationComponents:
    """Test visualization components in headless mode."""
    
    @pytest.fixture(autouse=True)
    def setup_pygame(self):
        """Set up pygame in headless mode for testing."""
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver
        pygame.init()
        yield
        pygame.quit()
    
    def test_grid_world_visualizer_init(self):
        """Test GridWorldVisualizer initialization."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        # Test initialization
        viz = GridWorldVisualizer(brainstem, cell_size=20, ui_width=200)
        
        assert viz.brainstem == brainstem
        assert viz.cell_size == 20
        assert viz.ui_width == 200
        assert viz.grid_width == 200  # 10 * 20
        assert viz.grid_height == 200  # 10 * 20
        
        # Test color mapping
        assert viz.get_cell_color(viz.simulation.EMPTY) == viz.COLORS['empty']
        assert viz.get_cell_color(viz.simulation.WALL) == viz.COLORS['wall']
        assert viz.get_cell_color(viz.simulation.FOOD) == viz.COLORS['food']
        assert viz.get_cell_color(viz.simulation.DANGER) == viz.COLORS['danger']
        
        viz.cleanup()
    
    def test_brain_state_monitor_init(self):
        """Test BrainStateMonitor initialization."""
        monitor = BrainStateMonitor(width=300, height=500)
        
        assert monitor.width == 300
        assert monitor.height == 500
        assert monitor.current_graph is None
        assert monitor.current_prediction_error == 0.0
        assert monitor.current_step == 0
        assert len(monitor.graph_stats_history) == 0
        assert len(monitor.prediction_history) == 0
        assert len(monitor.learning_events) == 0
    
    def test_brain_monitor_update(self):
        """Test brain monitor data updates."""
        monitor = BrainStateMonitor()
        graph = WorldGraph()
        
        # Create test experience nodes
        for i in range(3):
            node = ExperienceNode(
                mental_context=[float(i), float(i+1)],
                action_taken={"motor": float(i)},
                predicted_sensory=[float(i)] * 5,
                actual_sensory=[float(i+0.1)] * 5,
                prediction_error=0.1 * i
            )
            graph.add_node(node)
        
        # Update monitor
        action = {"forward_motor": 0.5, "turn_motor": 0.0}
        monitor.update(graph, prediction_error=0.3, recent_action=action, step=10)
        
        assert monitor.current_graph == graph
        assert monitor.current_prediction_error == 0.3
        assert monitor.current_step == 10
        assert len(monitor.graph_stats_history) == 1
        assert len(monitor.prediction_history) == 1
        assert len(monitor.learning_events) > 0  # Should have logged the action
        
        # Check stats
        stats = monitor.graph_stats_history[0]
        assert stats['total_nodes'] == 3
        assert stats['step'] == 10
        assert stats['prediction_error'] == 0.3
    
    def test_brain_monitor_event_logging(self):
        """Test event logging in brain monitor."""
        monitor = BrainStateMonitor()
        
        # Test manual event logging
        monitor._log_event("Test event", "info")
        monitor._log_event("Warning event", "warning")
        monitor._log_event("Success event", "success")
        
        assert len(monitor.learning_events) == 3
        
        # Check event structure
        event = monitor.learning_events[0]
        assert 'message' in event
        assert 'type' in event
        assert 'timestamp' in event
        assert 'step' in event
        
        # Test automatic event logging during update
        graph = WorldGraph()
        monitor.update(graph, prediction_error=2.5)  # High error should trigger warning
        
        # Should have added a warning event
        warning_events = [e for e in monitor.learning_events if e['type'] == 'warning']
        assert len(warning_events) > 0
    
    def test_integrated_display_init(self):
        """Test IntegratedDisplay initialization."""
        brainstem = GridWorldBrainstem(world_width=8, world_height=8, seed=42)
        
        display = IntegratedDisplay(brainstem, cell_size=30)
        
        assert display.brainstem == brainstem
        assert display.brain_graph is None
        assert display.running == True
        assert display.paused == False
        assert display.step_mode == False
        assert display.fps == 30
        
        # Test window dimensions
        expected_grid_width = 8 * 30  # 240
        expected_total_width = expected_grid_width + 400  # brain panel width
        
        assert display.window_width == expected_total_width
        
        display.cleanup()
    
    def test_integrated_display_brain_graph(self):
        """Test setting brain graph in integrated display."""
        brainstem = GridWorldBrainstem(world_width=5, world_height=5, seed=42)
        display = IntegratedDisplay(brainstem)
        
        # Create and set brain graph
        graph = WorldGraph()
        node = ExperienceNode(
            mental_context=[1.0, 2.0],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0] * 10,
            actual_sensory=[1.1] * 10,
            prediction_error=0.1
        )
        graph.add_node(node)
        
        display.set_brain_graph(graph)
        assert display.brain_graph == graph
        
        display.cleanup()
    
    def test_action_generation(self):
        """Test different action generation methods."""
        brainstem = GridWorldBrainstem(world_width=5, world_height=5, seed=42)
        display = IntegratedDisplay(brainstem)
        
        # Test default random action generation
        sensor_packet = brainstem.get_sensor_readings()
        action = display._generate_motor_action(sensor_packet)
        
        assert isinstance(action, dict)
        assert 'forward_motor' in action
        assert 'turn_motor' in action
        assert 'brake_motor' in action
        assert -1.0 <= action['forward_motor'] <= 1.0
        assert -1.0 <= action['turn_motor'] <= 1.0
        assert 0.0 <= action['brake_motor'] <= 1.0
        
        # Test action generator callback
        def test_generator():
            return {'forward_motor': 0.5, 'turn_motor': 0.0, 'brake_motor': 0.0}
        
        display.set_action_generator(test_generator)
        action = display._generate_motor_action(sensor_packet)
        
        assert action == {'forward_motor': 0.5, 'turn_motor': 0.0, 'brake_motor': 0.0}
        
        # Test learning callback
        def test_learning_callback(state):
            return {'forward_motor': 0.3, 'turn_motor': 0.1, 'brake_motor': 0.0}
        
        display.set_learning_callback(test_learning_callback)
        action = display._generate_motor_action(sensor_packet)
        
        assert action == {'forward_motor': 0.3, 'turn_motor': 0.1, 'brake_motor': 0.0}
        
        display.cleanup()
    
    def test_prediction_error_calculation(self):
        """Test prediction error calculation."""
        brainstem = GridWorldBrainstem(world_width=5, world_height=5, seed=42)
        display = IntegratedDisplay(brainstem)
        
        # Create mock sensor packets
        from core.communication import SensoryPacket
        from datetime import datetime
        
        old_packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        new_packet = SensoryPacket(
            sensor_values=[1.1, 2.1, 3.1],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=2
        )
        
        error = display._calculate_prediction_error(old_packet, new_packet)
        
        # Should be small positive value
        assert 0.0 <= error <= 1.0
        
        # Test identical packets (no error)
        identical_packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=3
        )
        
        zero_error = display._calculate_prediction_error(old_packet, identical_packet)
        assert zero_error == 0.0
        
        display.cleanup()
    
    def test_experience_node_creation(self):
        """Test experience node creation during simulation."""
        brainstem = GridWorldBrainstem(world_width=5, world_height=5, seed=42)
        display = IntegratedDisplay(brainstem)
        
        # Set up brain graph
        graph = WorldGraph()
        display.set_brain_graph(graph)
        
        # Create test data
        from core.communication import SensoryPacket
        from datetime import datetime
        
        sensor_packet = SensoryPacket(
            sensor_values=[1.0] * 22,  # 22 sensors
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        new_sensor_packet = SensoryPacket(
            sensor_values=[1.1] * 22,
            actuator_positions=[0.1, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=2
        )
        
        motor_commands = {"forward_motor": 0.5, "turn_motor": 0.0, "brake_motor": 0.0}
        prediction_error = 0.2
        
        # Create experience node
        initial_count = graph.node_count()
        display._create_experience_node(sensor_packet, motor_commands, new_sensor_packet, prediction_error)
        
        # Should have added one node
        assert graph.node_count() == initial_count + 1
        
        # Check node properties
        latest_node = graph.get_latest_node()
        assert latest_node is not None
        assert len(latest_node.mental_context) == 8  # First 8 sensors
        assert latest_node.action_taken == motor_commands
        assert latest_node.prediction_error == prediction_error
        
        display.cleanup()
    
    def test_simulation_state_reporting(self):
        """Test simulation state reporting."""
        brainstem = GridWorldBrainstem(world_width=5, world_height=5, seed=42)
        display = IntegratedDisplay(brainstem)
        
        # Set up brain graph
        graph = WorldGraph()
        display.set_brain_graph(graph)
        
        # Get simulation state
        state = display.get_simulation_state()
        
        assert 'brainstem_stats' in state
        assert 'world_state' in state
        assert 'brain_nodes' in state
        assert 'brain_stats' in state
        assert 'running' in state
        assert 'paused' in state
        assert 'fps' in state
        
        # Check values
        assert state['brain_nodes'] == 0  # Empty graph
        assert state['running'] == True
        assert state['paused'] == False
        assert isinstance(state['fps'], (int, float))
        
        display.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])