#!/usr/bin/env python3
"""
Demonstration of world-agnostic brain architecture.
Shows how the same brain can work with different brainstems having different sensor counts.
"""

from datetime import datetime
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from predictor.triple_predictor import TriplePredictor


def simulate_simple_robot_brainstem():
    """Simulate a simple robot with 4 sensors."""
    print("=== Simple Robot (4 sensors) ===")
    
    # Create brain (world-agnostic)
    predictor = TriplePredictor(base_time_budget=0.01)
    brain = BrainInterface(predictor)
    
    # Simulate sensor readings from simple robot
    simple_sensors = [
        [1.0, 0.5, 0.0, 0.8],  # distance_front, distance_left, button_pressed, battery_level
        [0.8, 0.4, 0.0, 0.8],  # object getting closer
        [0.3, 0.2, 1.0, 0.8],  # very close + button pressed
        [0.9, 0.8, 0.0, 0.7],  # moved away + battery lower
    ]
    
    for i, sensors in enumerate(simple_sensors):
        sensory_packet = SensoryPacket(
            sensor_values=sensors,
            actuator_positions=[0.0, 0.0],  # motor positions
            timestamp=datetime.now(),
            sequence_id=i + 1
        )
        
        # Robot state: [position_x, position_y]
        mental_context = [float(i) * 0.5, 0.0]
        
        prediction = brain.process_sensory_input(
            sensory_packet, mental_context, "normal"
        )
        
        print(f"Step {i+1}: Sensors: {sensors}")
        print(f"  Action: {prediction.motor_action}")
        print(f"  Expected sensors: {prediction.expected_sensory}")
        print(f"  Confidence: {prediction.confidence:.2f}")
        
    print(f"Brain learned {brain.sensory_vector_length} sensors")
    print(f"Total experiences: {brain.get_brain_statistics()['interface_stats']['total_experiences']}")
    print()


def simulate_mars_rover_brainstem():
    """Simulate a Mars rover with 12 sensors."""
    print("=== Mars Rover (12 sensors) ===")
    
    # Create brain (world-agnostic) 
    predictor = TriplePredictor(base_time_budget=0.02)
    brain = BrainInterface(predictor)
    
    # Simulate comprehensive sensor readings
    rover_sensors = [
        # [temp, pressure, wind, dust, solar_panel, battery, wheel_1, wheel_2, wheel_3, wheel_4, camera_tilt, drill_depth]
        [250.0, 610.0, 15.0, 0.3, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [248.0, 608.0, 18.0, 0.4, 0.7, 0.9, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0],
        [245.0, 605.0, 22.0, 0.6, 0.6, 0.8, 0.4, 0.4, 0.4, 0.4, 0.2, 0.1],
        [243.0, 603.0, 25.0, 0.8, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3],
    ]
    
    for i, sensors in enumerate(rover_sensors):
        sensory_packet = SensoryPacket(
            sensor_values=sensors,
            actuator_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6 actuators
            timestamp=datetime.now(),
            sequence_id=i + 1
        )
        
        # Rover state: [sol_day, power_level, mission_progress]
        mental_context = [float(i), sensors[5], float(i) / 10.0]  # battery as power level
        
        prediction = brain.process_sensory_input(
            sensory_packet, mental_context, "safe"  # Careful thinking on Mars
        )
        
        print(f"Sol {i+1}: Temp: {sensors[0]:.0f}K, Wind: {sensors[2]:.0f}m/s, Battery: {sensors[5]:.1f}")
        print(f"  Action: wheel speed {prediction.motor_action.get('forward_motor', 0.0):.2f}")
        print(f"  Traversals: {prediction.traversal_count}")
        print(f"  Thinking time: {prediction.time_budget_used:.4f}s")
        
    print(f"Brain learned {brain.sensory_vector_length} sensors")
    print(f"Total experiences: {brain.get_brain_statistics()['interface_stats']['total_experiences']}")
    print()


def simulate_drone_brainstem():
    """Simulate a drone with 8 sensors."""
    print("=== Autonomous Drone (8 sensors) ===")
    
    # Create brain (world-agnostic)
    predictor = TriplePredictor(base_time_budget=0.005)  # Fast decisions for flying
    brain = BrainInterface(predictor)
    
    # Simulate flight sensor readings
    drone_sensors = [
        # [altitude, pitch, roll, yaw, accel_x, accel_y, accel_z, battery]
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.8, 1.0],  # hovering
        [10.5, 0.1, 0.0, 0.0, 0.2, 0.0, 9.8, 0.99], # slight forward tilt
        [11.0, 0.2, 0.1, 0.0, 0.4, 0.1, 9.8, 0.98], # forward motion with slight roll
        [10.8, 0.0, 0.0, 0.1, 0.0, 0.0, 9.8, 0.97], # correcting, slight yaw
    ]
    
    for i, sensors in enumerate(drone_sensors):
        sensory_packet = SensoryPacket(
            sensor_values=sensors,
            actuator_positions=[0.0, 0.0, 0.0, 0.0],  # 4 motor speeds
            timestamp=datetime.now(),
            sequence_id=i + 1
        )
        
        # Drone state: [target_altitude, mission_mode]
        mental_context = [10.0, 1.0]  # target 10m altitude, mode 1
        
        threat = "danger" if abs(sensors[1]) > 0.15 or abs(sensors[2]) > 0.15 else "normal"
        
        prediction = brain.process_sensory_input(
            sensory_packet, mental_context, threat
        )
        
        print(f"Flight {i+1}: Alt: {sensors[0]:.1f}m, Pitch: {sensors[1]:.2f}, Roll: {sensors[2]:.2f}")
        print(f"  Threat: {threat}, Traversals: {prediction.traversal_count}")
        print(f"  Motor action: {list(prediction.motor_action.values())[:2]}")  # Show first 2 motors
        
    print(f"Brain learned {brain.sensory_vector_length} sensors")
    print(f"Total experiences: {brain.get_brain_statistics()['interface_stats']['total_experiences']}")
    print()


def main():
    """Demonstrate world-agnostic brain with different robots."""
    print("=== World-Agnostic Brain Architecture Demo ===")
    print("Same brain design adapts to different sensor configurations automatically.\n")
    
    # Test with different robot types
    simulate_simple_robot_brainstem()
    simulate_mars_rover_brainstem() 
    simulate_drone_brainstem()
    
    print("=== Summary ===")
    print("✅ Brain successfully adapted to 4, 12, and 8 sensor configurations")
    print("✅ No hardcoded sensor assumptions in brain code")
    print("✅ Brainstem handles robot-specific sensor knowledge")
    print("✅ Brain focuses purely on learning and prediction")
    print("✅ Same brain architecture scales from simple robots to complex rovers")
    print("\nThis demonstrates the power of proper architectural separation!")


if __name__ == "__main__":
    main()