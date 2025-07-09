"""
Brainstem simulation interface that mimics the real brainstem communication protocol.
Bridges the grid world simulation with the brain's expected interface via socket communication.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from .grid_world import GridWorldSimulation
from core.communication import PredictionPacket, SensoryPacket
from network.brain_client import BrainSocketClient


class GridWorldBrainstem:
    """
    Simulation interface that mimics the real brainstem communication protocol.
    Provides sensor readings and executes motor commands in the grid world via socket communication.
    """
    
    def __init__(self, world_width: int = 40, world_height: int = 40, seed: int = None,
                 brain_host: str = "localhost", brain_port: int = 8080, use_sockets: bool = True):
        """
        Initialize the brainstem simulation.
        
        Args:
            world_width: Grid world width
            world_height: Grid world height  
            seed: Random seed for world generation
            brain_host: Brain server hostname
            brain_port: Brain server port
            use_sockets: Whether to use socket communication (True) or local brain (False)
        """
        self.simulation = GridWorldSimulation(world_width, world_height, seed)
        self.sequence_counter = 0
        self.last_motor_commands = {"forward_motor": 0.0, "turn_motor": 0.0, "brake_motor": 0.0}
        
        # Socket communication
        self.use_sockets = use_sockets
        if use_sockets:
            self.brain_client = BrainSocketClient(
                brain_host, brain_port, 
                client_name=f"GridWorld-{world_width}x{world_height}"
            )
        else:
            # Fallback to local brain for backwards compatibility
            from core.brain_interface import BrainInterface
            from predictor.multi_drive_predictor import MultiDrivePredictor
            # Use the complete brain system with all capabilities and GPU acceleration
            predictor = MultiDrivePredictor(base_time_budget=0.1)
            self.brain_client = BrainInterface(predictor, enable_persistence=True, use_gpu=True)
        
    def get_sensor_readings(self) -> SensoryPacket:
        """
        Return sensor data in the same format as real brainstem.
        Compatible with brain's SensoryPacket structure.
        """
        sensor_values = self.simulation.get_sensor_readings()
        
        # Current actuator positions (simulated)
        actuator_positions = [
            self.last_motor_commands["forward_motor"],
            self.last_motor_commands["turn_motor"], 
            self.last_motor_commands["brake_motor"]
        ]
        
        self.sequence_counter += 1
        
        return SensoryPacket(
            sensor_values=sensor_values,
            actuator_positions=actuator_positions,
            timestamp=datetime.now(),
            sequence_id=self.sequence_counter,
            network_latency=0.001  # Simulated minimal latency
        )
    
    def execute_motor_commands(self, commands: Dict[str, float]) -> bool:
        """
        Process motor commands and update simulation state.
        Returns True if robot is still alive, False if terminated.
        """
        # Store commands for next sensor reading
        self.last_motor_commands.update(commands)
        
        # Execute in simulation
        is_alive = self.simulation.execute_motor_commands(commands)
        
        if not is_alive:
            print(f"Robot died at step {self.simulation.step_count}!")
            self.reset_robot()
            return False
        
        return True
    
    def execute_prediction_packet(self, prediction: PredictionPacket) -> bool:
        """
        Execute a prediction packet's motor commands.
        Returns True if robot is still alive.
        """
        return self.execute_motor_commands(prediction.motor_action)
    
    def reset_robot(self):
        """Reset robot to safe starting position with full health/energy."""
        self.simulation.reset_robot()
        self.sequence_counter = 0
        self.last_motor_commands = {"forward_motor": 0.0, "turn_motor": 0.0, "brake_motor": 0.0}
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """
        Mimic real brainstem hardware discovery.
        Returns sensor and actuator specifications.
        """
        return {
            'sensors': [
                {'id': 'distance_sensors', 'type': 'distance', 'data_size': 4},
                {'id': 'vision_features', 'type': 'camera_features', 'data_size': 13},
                {'id': 'smell_sensors', 'type': 'chemical', 'data_size': 2},
                {'id': 'internal_state', 'type': 'internal', 'data_size': 5}
            ],
            'actuators': [
                {'id': 'forward_motor', 'type': 'motor', 'range': [-1.0, 1.0]},
                {'id': 'turn_motor', 'type': 'motor', 'range': [-1.0, 1.0]},
                {'id': 'brake_motor', 'type': 'motor', 'range': [0.0, 1.0]}
            ],
            'total_sensor_size': 24,
            'simulation_mode': True
        }
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get detailed simulation statistics."""
        stats = self.simulation.get_simulation_stats()
        stats.update({
            'sequence_counter': self.sequence_counter,
            'last_motor_commands': self.last_motor_commands.copy()
        })
        return stats
    
    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state for visualization or analysis."""
        world_copy = self.simulation.get_world_copy()
        return {
            'world_grid': world_copy.tolist(),
            'robot_position': self.simulation.robot.position,
            'robot_orientation': self.simulation.robot.orientation,
            'robot_health': self.simulation.robot.health,
            'robot_energy': self.simulation.robot.energy,
            'world_size': (self.simulation.width, self.simulation.height)
        }
    
    def is_robot_alive(self) -> bool:
        """Check if robot is still alive."""
        return self.simulation.robot.health > 0.0
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get information about sensor layout and meanings."""
        return {
            'sensor_layout': {
                'distance_sensors': {
                    'indices': [0, 1, 2, 3],
                    'meanings': ['front_distance', 'left_distance', 'right_distance', 'back_distance'],
                    'range': [0.0, 1.0],
                    'description': 'Distance to obstacles (0.0=adjacent, 1.0=max_range)'
                },
                'vision_features': {
                    'indices': list(range(4, 17)),
                    'meanings': [
                        'cell_nw', 'cell_n', 'cell_ne', 'cell_w', 'cell_center', 'cell_e', 
                        'cell_sw', 'cell_s', 'cell_se', 'nearest_food_direction',
                        'nearest_danger_direction', 'food_density', 'danger_density'
                    ],
                    'range': [0.0, 1.0],
                    'description': '3x3 grid around robot + environmental features'
                },
                'smell_sensors': {
                    'indices': [17, 18],
                    'meanings': ['plant_direction', 'plant_intensity'],
                    'range': [0.0, 1.0],
                    'description': 'Chemical/scent detection sensors'
                },
                'internal_state': {
                    'indices': [19, 20, 21, 22, 23],
                    'meanings': ['health', 'energy', 'orientation', 'time_since_food', 'time_since_damage'],
                    'range': [0.0, 1.0],
                    'description': 'Robot internal sensors'
                }
            },
            'total_sensors': 24,
            'cell_values': {
                0.0: 'Empty',
                1.0: 'Wall', 
                0.5: 'Food',
                -1.0: 'Danger',
                0.8: 'Robot'
            }
        }
    
    def step(self, motor_commands: Dict[str, float] = None) -> SensoryPacket:
        """
        Perform one simulation step.
        If motor_commands provided, execute them first, then return new sensor reading.
        """
        if motor_commands:
            self.execute_motor_commands(motor_commands)
        
        return self.get_sensor_readings()
    
    def run_steps(self, num_steps: int, motor_commands: Dict[str, float] = None) -> List[SensoryPacket]:
        """
        Run multiple simulation steps with the same motor commands.
        Returns list of sensor readings from each step.
        """
        readings = []
        for _ in range(num_steps):
            readings.append(self.step(motor_commands))
            if not self.is_robot_alive():
                break
        return readings
    
    async def connect_to_brain(self) -> bool:
        """Connect to brain server (if using sockets)."""
        if self.use_sockets:
            return await self.brain_client.connect()
        return True  # Local brain is always "connected"
    
    async def disconnect_from_brain(self):
        """Disconnect from brain server (if using sockets)."""
        if self.use_sockets:
            await self.brain_client.disconnect()
    
    async def get_brain_prediction(self, mental_context: List[float], 
                                 threat_level: str = "normal") -> Optional[PredictionPacket]:
        """
        Get prediction from brain using current sensor readings.
        
        Args:
            mental_context: Current mental state (position, energy, etc.)
            threat_level: Threat assessment for time budgeting
            
        Returns:
            Prediction packet from brain or None if communication failed
        """
        sensory_packet = self.get_sensor_readings()
        
        if self.use_sockets:
            return await self.brain_client.process_sensory_input(
                sensory_packet, mental_context, threat_level
            )
        else:
            # Local brain interface
            return self.brain_client.process_sensory_input(
                sensory_packet, mental_context, threat_level
            )
    
    async def run_brain_controlled_simulation(self, steps: int = 100, 
                                            step_delay: float = 0.1) -> Dict[str, Any]:
        """
        Run simulation with brain making all decisions.
        
        Args:
            steps: Number of simulation steps to run
            step_delay: Delay between steps (seconds)
            
        Returns:
            Dictionary with simulation results and statistics
        """
        print(f"Starting brain-controlled simulation for {steps} steps...")
        
        if not await self.connect_to_brain():
            raise RuntimeError("Failed to connect to brain server")
        
        try:
            results = {
                "steps_completed": 0,
                "predictions_received": 0,
                "communication_errors": 0,
                "final_robot_state": None,
                "performance_stats": []
            }
            
            for step in range(steps):
                try:
                    # Calculate mental context (robot's understanding of its state)
                    robot_pos = self.simulation.robot.position
                    mental_context = [
                        float(robot_pos[0]) / self.simulation.width,   # normalized x
                        float(robot_pos[1]) / self.simulation.height, # normalized y
                        self.simulation.robot.energy,                 # energy level
                        self.simulation.robot.health,                 # health level
                        float(step) / steps                           # mission progress
                    ]
                    
                    # Determine threat level based on robot state
                    if self.simulation.robot.health < 0.3 or self.simulation.robot.energy < 0.2:
                        threat = "danger"
                    elif self.simulation.robot.health < 0.6 or self.simulation.robot.energy < 0.5:
                        threat = "alert"
                    else:
                        threat = "normal"
                    
                    # Get prediction from brain
                    prediction = await self.get_brain_prediction(mental_context, threat)
                    
                    if prediction:
                        results["predictions_received"] += 1
                        
                        # Execute brain's decision
                        is_alive = self.execute_prediction_packet(prediction)
                        
                        # Log performance data
                        results["performance_stats"].append({
                            "step": step,
                            "threat_level": threat,
                            "consensus_strength": prediction.consensus_strength,
                            "traversal_count": prediction.traversal_count,
                            "confidence": prediction.confidence,
                            "robot_health": self.simulation.robot.health,
                            "robot_energy": self.simulation.robot.energy
                        })
                        
                        if not is_alive:
                            print(f"Robot died at step {step}")
                            break
                    else:
                        results["communication_errors"] += 1
                        print(f"Failed to get brain prediction at step {step}")
                        # Emergency stop
                        self.execute_motor_commands({"forward_motor": 0.0, "turn_motor": 0.0, "brake_motor": 1.0})
                    
                    results["steps_completed"] = step + 1
                    
                    # Progress reporting
                    if step % 10 == 0:
                        print(f"Step {step}: Health={self.simulation.robot.health:.2f}, "
                              f"Energy={self.simulation.robot.energy:.2f}, Threat={threat}")
                    
                    # Delay between steps
                    if step_delay > 0:
                        await asyncio.sleep(step_delay)
                        
                except Exception as e:
                    print(f"Error at step {step}: {e}")
                    results["communication_errors"] += 1
                    break
            
            # Final state
            results["final_robot_state"] = {
                "position": self.simulation.robot.position,
                "health": self.simulation.robot.health,
                "energy": self.simulation.robot.energy,
                "alive": self.is_robot_alive()
            }
            
            print(f"Simulation completed: {results['steps_completed']} steps, "
                  f"{results['predictions_received']} predictions, "
                  f"{results['communication_errors']} errors")
            
            return results
            
        finally:
            await self.disconnect_from_brain()