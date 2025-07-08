"""
Unit tests for grid world simulation.
"""

import pytest
import numpy as np
from simulation.grid_world import GridWorldSimulation, Robot
from simulation.brainstem_sim import GridWorldBrainstem


class TestGridWorldSimulation:
    """Test cases for GridWorldSimulation functionality."""
    
    def test_simulation_initialization(self):
        """Test basic simulation initialization."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        assert sim.width == 10
        assert sim.height == 10
        assert sim.world.shape == (10, 10)
        assert sim.step_count == 0
        assert sim.robot.health == 1.0
        assert sim.robot.energy == 1.0
        
        # Check perimeter walls
        assert np.all(sim.world[0, :] == sim.WALL)   # Top
        assert np.all(sim.world[-1, :] == sim.WALL)  # Bottom
        assert np.all(sim.world[:, 0] == sim.WALL)   # Left
        assert np.all(sim.world[:, -1] == sim.WALL)  # Right
    
    def test_robot_positioning(self):
        """Test robot positioning and movement."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        initial_pos = sim.robot.position
        assert 1 <= initial_pos[0] <= 8  # Not on walls
        assert 1 <= initial_pos[1] <= 8  # Not on walls
        
        # Test position is in empty cell
        x, y = initial_pos
        assert sim.world[x, y] == sim.EMPTY
    
    def test_sensor_readings_format(self):
        """Test that sensor readings have correct format and length."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        sensors = sim.get_sensor_readings()
        
        # Should have exactly 22 sensor values
        assert len(sensors) == 22
        
        # All values should be floats
        assert all(isinstance(val, (int, float)) for val in sensors)
        
        # Distance sensors (first 4) should be 0-1
        distance_sensors = sensors[0:4]
        assert all(0.0 <= val <= 1.0 for val in distance_sensors)
        
        # Vision features (next 13) should be in valid ranges
        vision_features = sensors[4:17]
        assert len(vision_features) == 13
        
        # Internal state (last 5) should be 0-1
        internal_state = sensors[17:22]
        assert all(0.0 <= val <= 1.0 for val in internal_state)
    
    def test_motor_command_execution(self):
        """Test motor command execution."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        initial_pos = sim.robot.position
        initial_orientation = sim.robot.orientation
        
        # Test turning right
        result = sim.execute_motor_commands({"turn_motor": 0.5})
        assert result  # Robot should still be alive
        assert sim.robot.orientation == (initial_orientation + 1) % 4
        assert sim.robot.position == initial_pos  # Position shouldn't change
        
        # Test moving forward
        sim.robot.orientation = 1  # Face East
        result = sim.execute_motor_commands({"forward_motor": 0.5})
        assert result
        
        # Position should change (unless blocked)
        new_pos = sim.robot.position
        expected_pos = (initial_pos[0] + 1, initial_pos[1])
        
        # Check if movement was successful or blocked
        if sim.world[expected_pos[0], expected_pos[1]] != sim.WALL:
            assert new_pos == expected_pos
        else:
            assert new_pos == initial_pos  # Blocked by wall
    
    def test_energy_and_health_mechanics(self):
        """Test energy consumption and health mechanics."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        initial_energy = sim.robot.energy
        
        # Execute motor command that should consume energy
        sim.execute_motor_commands({"forward_motor": 1.0})
        
        # Energy should decrease
        assert sim.robot.energy < initial_energy
        
        # Test food consumption if possible
        x, y = sim.robot.position
        if sim.world[x, y] == sim.FOOD:
            initial_energy = sim.robot.energy
            sim._process_environment_effects()
            assert sim.robot.energy > initial_energy  # Energy restored
            assert sim.world[x, y] == sim.EMPTY  # Food consumed
    
    def test_collision_handling(self):
        """Test collision detection and damage."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        # Place robot next to a wall
        sim.robot.position = (1, 1)
        sim.robot.orientation = 3  # Face West (toward wall)
        
        initial_health = sim.robot.health
        initial_collisions = sim.total_collisions
        
        # Try to move into wall
        sim.execute_motor_commands({"forward_motor": 0.5})
        
        # Should take damage and not move
        assert sim.robot.health < initial_health
        assert sim.total_collisions > initial_collisions
        assert sim.robot.position == (1, 1)  # Didn't move into wall
    
    def test_simulation_statistics(self):
        """Test simulation statistics collection."""
        sim = GridWorldSimulation(width=10, height=10, seed=42)
        
        stats = sim.get_simulation_stats()
        
        required_keys = [
            "step_count", "robot_health", "robot_energy", "robot_position",
            "robot_orientation", "total_food_consumed", "total_collisions",
            "time_since_food", "time_since_damage"
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats["step_count"], int)
        assert isinstance(stats["robot_health"], float)
        assert isinstance(stats["robot_energy"], float)
        assert isinstance(stats["robot_position"], tuple)


class TestGridWorldBrainstem:
    """Test cases for GridWorldBrainstem interface."""
    
    def test_brainstem_initialization(self):
        """Test brainstem interface initialization."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        assert brainstem.simulation.width == 10
        assert brainstem.simulation.height == 10
        assert brainstem.sequence_counter == 0
    
    def test_sensor_packet_format(self):
        """Test that sensor packets have correct format."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        packet = brainstem.get_sensor_readings()
        
        assert len(packet.sensor_values) == 22
        assert len(packet.actuator_positions) == 3
        assert isinstance(packet.timestamp, type(packet.timestamp))
        assert isinstance(packet.sequence_id, int)
        assert packet.sequence_id == 1  # First reading
        
        # Second reading should increment sequence
        packet2 = brainstem.get_sensor_readings()
        assert packet2.sequence_id == 2
    
    def test_motor_command_execution(self):
        """Test motor command execution through brainstem."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        initial_stats = brainstem.get_simulation_stats()
        
        # Execute motor commands
        result = brainstem.execute_motor_commands({
            "forward_motor": 0.3,
            "turn_motor": 0.0,
            "brake_motor": 0.0
        })
        
        assert result  # Robot should still be alive
        
        new_stats = brainstem.get_simulation_stats()
        assert new_stats["step_count"] > initial_stats["step_count"]
    
    def test_hardware_capabilities(self):
        """Test hardware capabilities reporting."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        capabilities = brainstem.get_hardware_capabilities()
        
        assert "sensors" in capabilities
        assert "actuators" in capabilities
        assert capabilities["total_sensor_size"] == 22
        assert capabilities["simulation_mode"] is True
        
        # Check sensor specifications
        sensors = capabilities["sensors"]
        assert len(sensors) == 3
        assert sum(sensor["data_size"] for sensor in sensors) == 22
        
        # Check actuator specifications
        actuators = capabilities["actuators"]
        assert len(actuators) == 3
        motor_names = [act["id"] for act in actuators]
        assert "forward_motor" in motor_names
        assert "turn_motor" in motor_names
        assert "brake_motor" in motor_names
    
    def test_world_state_reporting(self):
        """Test world state reporting for visualization."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        world_state = brainstem.get_world_state()
        
        assert "world_grid" in world_state
        assert "robot_position" in world_state
        assert "robot_orientation" in world_state
        assert "robot_health" in world_state
        assert "robot_energy" in world_state
        assert "world_size" in world_state
        
        assert world_state["world_size"] == (10, 10)
        assert len(world_state["world_grid"]) == 10
        assert len(world_state["world_grid"][0]) == 10
    
    def test_sensor_info(self):
        """Test sensor information reporting."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        sensor_info = brainstem.get_sensor_info()
        
        assert "sensor_layout" in sensor_info
        assert "total_sensors" in sensor_info
        assert sensor_info["total_sensors"] == 22
        
        layout = sensor_info["sensor_layout"]
        assert "distance_sensors" in layout
        assert "vision_features" in layout
        assert "internal_state" in layout
        
        # Check index ranges don't overlap and cover all sensors
        distance_indices = layout["distance_sensors"]["indices"]
        vision_indices = layout["vision_features"]["indices"]
        internal_indices = layout["internal_state"]["indices"]
        
        all_indices = distance_indices + vision_indices + internal_indices
        assert len(all_indices) == 22
        assert len(set(all_indices)) == 22  # No duplicates
        assert min(all_indices) == 0
        assert max(all_indices) == 21
    
    def test_step_function(self):
        """Test single step execution."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        # Step without commands
        packet1 = brainstem.step()
        assert len(packet1.sensor_values) == 22
        
        # Step with commands
        packet2 = brainstem.step({"forward_motor": 0.2})
        assert len(packet2.sensor_values) == 22
        assert packet2.sequence_id > packet1.sequence_id
    
    def test_multiple_steps(self):
        """Test running multiple steps."""
        brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=42, use_sockets=False)
        
        readings = brainstem.run_steps(5, {"forward_motor": 0.1})
        
        assert len(readings) <= 5  # Could be less if robot dies
        assert all(len(reading.sensor_values) == 22 for reading in readings)
        
        # Sequence IDs should be increasing
        for i in range(1, len(readings)):
            assert readings[i].sequence_id > readings[i-1].sequence_id


if __name__ == "__main__":
    pytest.main([__file__])