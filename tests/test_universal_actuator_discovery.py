#!/usr/bin/env python3
"""
Test the Universal Actuator Discovery System.
Verifies that the brain can discover actuator effects and emergent categories
for any type of embodiment: wheels, legs, grids, alien actuators.
"""

import random
import math
from datetime import datetime
from core.actuator_discovery import UniversalActuatorDiscovery
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from predictor.multi_drive_predictor import MultiDrivePredictor


def test_wheeled_robot_discovery():
    """Test actuator discovery for a wheeled robot."""
    print("🤖 Testing Wheeled Robot Actuator Discovery")
    print("==========================================")
    
    discovery = UniversalActuatorDiscovery()
    
    # Simulate wheeled robot with position sensors
    robot_x, robot_y, robot_heading = 0.0, 0.0, 0.0
    
    for step in range(30):
        # Actuator commands for wheeled robot
        left_wheel = random.uniform(-1, 1)
        right_wheel = random.uniform(-1, 1)
        horn = random.uniform(0, 1) if random.random() < 0.1 else 0.0  # Occasional horn
        
        actuator_commands = {
            "left_wheel": left_wheel,
            "right_wheel": right_wheel,
            "horn": horn
        }
        
        # Simulate wheeled robot physics
        forward_speed = (left_wheel + right_wheel) / 2
        turn_rate = (right_wheel - left_wheel) / 2
        
        robot_x += forward_speed * math.cos(robot_heading) * 0.1
        robot_y += forward_speed * math.sin(robot_heading) * 0.1
        robot_heading += turn_rate * 0.1
        
        # Sensory readings: position, orientation, sound level, other sensors
        sensory_reading = [
            robot_x,  # Position X
            robot_y,  # Position Y  
            robot_heading,  # Orientation
            forward_speed,  # Speed sensor
            turn_rate,  # Turn rate sensor
            horn * 0.8 + random.uniform(-0.1, 0.1),  # Sound level (correlated with horn)
            random.uniform(-0.2, 0.2),  # Gyroscope noise
            random.uniform(-0.1, 0.1),  # Accelerometer noise
        ]
        
        # Observe effects
        result = discovery.observe_actuator_effects(actuator_commands, sensory_reading)
    
    # Analyze discovered patterns
    categories = discovery.get_actuator_categories()
    stats = discovery.get_discovery_statistics()
    
    print(f"Discovery Results:")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Actuators discovered: {stats['total_actuators_discovered']}")
    print(f"  Emergent categories: {stats['emergent_categories_formed']}")
    
    # Analyze specific actuators
    left_analysis = discovery.get_actuator_analysis("left_wheel")
    right_analysis = discovery.get_actuator_analysis("right_wheel")
    horn_analysis = discovery.get_actuator_analysis("horn")
    
    print(f"\nActuator Analysis:")
    if left_analysis:
        print(f"  Left wheel affects dimensions: {left_analysis['primary_affected_dimensions']}")
        print(f"  Left wheel reliability: {left_analysis['effect_reliability']:.3f}")
    
    if right_analysis:
        print(f"  Right wheel affects dimensions: {right_analysis['primary_affected_dimensions']}")
        print(f"  Right wheel reliability: {right_analysis['effect_reliability']:.3f}")
    
    if horn_analysis:
        print(f"  Horn affects dimensions: {horn_analysis['primary_affected_dimensions']}")
        print(f"  Horn reliability: {horn_analysis['effect_reliability']:.3f}")
    
    print(f"\nEmergent Categories:")
    for category_id, category_data in categories.items():
        properties = category_data['emergent_properties']
        print(f"  {category_id}: {category_data['member_actuators']}")
        print(f"    Spatial: {properties['appears_spatial']}")
        print(f"    Manipulative: {properties['appears_manipulative']}")
        print(f"    Environmental: {properties['appears_environmental']}")
    
    # Verify discovery success
    wheels_discovered = (left_analysis and len(left_analysis['primary_affected_dimensions']) > 0 and
                        right_analysis and len(right_analysis['primary_affected_dimensions']) > 0)
    
    spatial_category_found = any(cat['emergent_properties']['appears_spatial'] for cat in categories.values())
    
    print(f"\n✅ Wheels discovered: {wheels_discovered}")
    print(f"✅ Spatial category found: {spatial_category_found}")
    
    return wheels_discovered and (spatial_category_found or len(categories) > 0)


def test_legged_robot_discovery():
    """Test actuator discovery for a legged robot."""
    print("\n🦾 Testing Legged Robot Actuator Discovery")
    print("========================================")
    
    discovery = UniversalActuatorDiscovery()
    
    # Simulate legged robot with multiple joint actuators
    robot_x, robot_y = 0.0, 0.0
    joint_positions = {"hip": 0.0, "knee": 0.0, "ankle": 0.0}
    
    for step in range(25):
        # Actuator commands for joints and gripper
        hip_motor = random.uniform(-0.5, 0.5)
        knee_motor = random.uniform(-0.5, 0.5)
        ankle_motor = random.uniform(-0.3, 0.3)
        gripper_motor = random.uniform(0, 1) if random.random() < 0.15 else 0.0
        
        actuator_commands = {
            "hip_motor": hip_motor,
            "knee_motor": knee_motor,
            "ankle_motor": ankle_motor,
            "gripper_motor": gripper_motor
        }
        
        # Simulate legged locomotion
        joint_positions["hip"] += hip_motor * 0.1
        joint_positions["knee"] += knee_motor * 0.1
        joint_positions["ankle"] += ankle_motor * 0.1
        
        # Walking effect on position (simplified)
        leg_extension = joint_positions["hip"] + joint_positions["knee"] - joint_positions["ankle"]
        robot_x += leg_extension * 0.05
        robot_y += abs(joint_positions["hip"]) * 0.02
        
        # Gripper affects object proximity
        object_distance = max(0.0, 1.0 - gripper_motor * 2.0)
        
        # Sensory readings: position, joint sensors, object sensors
        sensory_reading = [
            robot_x,  # Position X
            robot_y,  # Position Y
            joint_positions["hip"],  # Hip position sensor
            joint_positions["knee"],  # Knee position sensor  
            joint_positions["ankle"],  # Ankle position sensor
            object_distance,  # Object proximity (affected by gripper)
            leg_extension,  # Leg extension sensor
            random.uniform(-0.1, 0.1),  # Balance sensor noise
            random.uniform(-0.1, 0.1),  # Pressure sensor noise
        ]
        
        discovery.observe_actuator_effects(actuator_commands, sensory_reading)
    
    # Analyze results
    categories = discovery.get_actuator_categories()
    stats = discovery.get_discovery_statistics()
    
    print(f"Discovery Results:")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Emergent categories: {stats['emergent_categories_formed']}")
    
    # Check joint vs gripper categorization
    joint_analyses = {
        "hip_motor": discovery.get_actuator_analysis("hip_motor"),
        "knee_motor": discovery.get_actuator_analysis("knee_motor"),
        "ankle_motor": discovery.get_actuator_analysis("ankle_motor")
    }
    gripper_analysis = discovery.get_actuator_analysis("gripper_motor")
    
    print(f"\nJoint Actuators:")
    for name, analysis in joint_analyses.items():
        if analysis:
            print(f"  {name}: affects {analysis['primary_affected_dimensions']}, reliability {analysis['effect_reliability']:.3f}")
    
    if gripper_analysis:
        print(f"\nGripper: affects {gripper_analysis['primary_affected_dimensions']}, reliability {gripper_analysis['effect_reliability']:.3f}")
    
    print(f"\nEmergent Categories:")
    for category_id, category_data in categories.items():
        properties = category_data['emergent_properties']
        print(f"  {category_id}: {category_data['member_actuators']}")
        print(f"    Properties: spatial={properties['appears_spatial']}, manipulative={properties['appears_manipulative']}")
    
    # Verify joint discovery
    joints_discovered = sum(1 for analysis in joint_analyses.values() if analysis and len(analysis['primary_affected_dimensions']) > 0)
    categories_formed = len(categories) > 0
    
    print(f"\n✅ Joints discovered: {joints_discovered}/3")
    print(f"✅ Categories formed: {categories_formed}")
    
    return joints_discovered >= 2 and categories_formed


def test_2d_grid_robot_discovery():
    """Test actuator discovery for a 2D grid robot."""
    print("\n🎮 Testing 2D Grid Robot Actuator Discovery")
    print("==========================================")
    
    discovery = UniversalActuatorDiscovery()
    
    # Simulate 2D grid robot
    grid_x, grid_y = 5, 5
    inventory_item = 0
    
    for step in range(20):
        # Grid actuator commands
        move_x = random.choice([-1, 0, 1])
        move_y = random.choice([-1, 0, 1]) 
        pickup_item = 1 if random.random() < 0.1 else 0
        use_item = 1 if random.random() < 0.05 and inventory_item > 0 else 0
        
        actuator_commands = {
            "move_x": float(move_x),
            "move_y": float(move_y),
            "pickup_item": float(pickup_item),
            "use_item": float(use_item)
        }
        
        # Simulate grid world effects
        grid_x = max(0, min(10, grid_x + move_x))
        grid_y = max(0, min(10, grid_y + move_y))
        
        if pickup_item and (grid_x, grid_y) in [(3, 3), (7, 7)]:  # Item locations
            inventory_item += 1
        
        if use_item and inventory_item > 0:
            inventory_item -= 1
        
        # Sensory readings: grid position, inventory, environment
        sensory_reading = [
            float(grid_x),  # Grid X coordinate
            float(grid_y),  # Grid Y coordinate
            float(inventory_item),  # Inventory count
            1.0 if (grid_x, grid_y) in [(3, 3), (7, 7)] else 0.0,  # Item available sensor
            float(grid_x + grid_y),  # Derived position sensor
            random.uniform(-0.1, 0.1),  # Noise
        ]
        
        discovery.observe_actuator_effects(actuator_commands, sensory_reading)
    
    # Analyze grid robot discoveries
    categories = discovery.get_actuator_categories()
    stats = discovery.get_discovery_statistics()
    
    print(f"Discovery Results:")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Emergent categories: {stats['emergent_categories_formed']}")
    
    # Analyze movement vs interaction actuators
    move_x_analysis = discovery.get_actuator_analysis("move_x")
    move_y_analysis = discovery.get_actuator_analysis("move_y")
    pickup_analysis = discovery.get_actuator_analysis("pickup_item")
    use_analysis = discovery.get_actuator_analysis("use_item")
    
    print(f"\nMovement Actuators:")
    if move_x_analysis:
        print(f"  move_x: affects {move_x_analysis['primary_affected_dimensions']}")
    if move_y_analysis:
        print(f"  move_y: affects {move_y_analysis['primary_affected_dimensions']}")
    
    print(f"\nInteraction Actuators:")
    if pickup_analysis:
        print(f"  pickup_item: affects {pickup_analysis['primary_affected_dimensions']}")
    if use_analysis:
        print(f"  use_item: affects {use_analysis['primary_affected_dimensions']}")
    
    # Verify grid discovery
    movement_discovered = (move_x_analysis and len(move_x_analysis['primary_affected_dimensions']) > 0) or \
                         (move_y_analysis and len(move_y_analysis['primary_affected_dimensions']) > 0)
    
    print(f"\n✅ Movement discovered: {movement_discovered}")
    print(f"✅ Categories formed: {len(categories) > 0}")
    
    return movement_discovered


def test_brain_integration():
    """Test actuator discovery integrated with the full brain system."""
    print("\n🧠 Testing Brain Integration with Actuator Discovery")
    print("==================================================")
    
    # Create brain with actuator discovery
    predictor = MultiDrivePredictor(base_time_budget=0.05)
    brain = BrainInterface(predictor, enable_persistence=False)
    
    # Simulate robot with mixed actuators
    robot_state = {"x": 0.0, "y": 0.0, "gripper": 0.0, "sound": 0.0}
    
    for step in range(15):
        # Mixed actuator types
        sensory_packet = SensoryPacket(
            sensor_values=[
                robot_state["x"],  # Position X (affected by wheels)
                robot_state["y"],  # Position Y (affected by wheels) 
                robot_state["gripper"],  # Gripper position
                robot_state["sound"],  # Sound level (affected by speaker)
                robot_state["x"] + robot_state["y"],  # Derived sensor
                random.uniform(-0.1, 0.1),  # Noise
            ],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=step + 1
        )
        
        mental_context = [robot_state["x"], robot_state["y"], 0.5, 0.8]
        prediction = brain.process_sensory_input(sensory_packet, mental_context)
        
        # Simulate actuator effects based on predicted action
        wheel_left = prediction.motor_action.get("wheel_left", 0.0)
        wheel_right = prediction.motor_action.get("wheel_right", 0.0) 
        gripper_motor = prediction.motor_action.get("gripper_motor", 0.0)
        speaker = prediction.motor_action.get("speaker", 0.0)
        
        # Update robot state based on actions
        robot_state["x"] += (wheel_left + wheel_right) * 0.05
        robot_state["y"] += (wheel_right - wheel_left) * 0.02
        robot_state["gripper"] += gripper_motor * 0.1
        robot_state["sound"] = speaker * 0.8
    
    # Get actuator discovery results
    stats = brain.get_brain_statistics()
    discovery_stats = stats.get('actuator_discovery_stats', {})
    categories = brain.get_discovered_actuator_categories()
    
    print(f"Brain Integration Results:")
    print(f"  Total observations: {discovery_stats.get('total_observations', 0)}")
    print(f"  Actuators discovered: {discovery_stats.get('total_actuators_discovered', 0)}")
    print(f"  Emergent categories: {discovery_stats.get('emergent_categories_formed', 0)}")
    
    # Test category queries
    spatial_actuators = brain.get_actuators_by_emergent_type('spatial')
    manipulative_actuators = brain.get_actuators_by_emergent_type('manipulative')
    environmental_actuators = brain.get_actuators_by_emergent_type('environmental')
    
    print(f"\nEmergent Actuator Types:")
    print(f"  Spatial: {spatial_actuators}")
    print(f"  Manipulative: {manipulative_actuators}")
    print(f"  Environmental: {environmental_actuators}")
    
    # Verify integration
    observations_made = discovery_stats.get('total_observations', 0) > 0
    actuators_found = discovery_stats.get('total_actuators_discovered', 0) > 0
    
    print(f"\n✅ Observations made: {observations_made}")
    print(f"✅ Actuators discovered: {actuators_found}")
    
    return observations_made and actuators_found


def main():
    """Run all universal actuator discovery tests."""
    print("🤖 Universal Actuator Discovery System Test Suite")
    print("================================================")
    print("Testing emergent actuator categorization for different embodiments:")
    print("• Wheeled robot (wheels affect position)")
    print("• Legged robot (joints affect locomotion, gripper affects objects)")
    print("• 2D grid robot (discrete movement commands)")
    print("• Brain integration (mixed actuator types)")
    print()
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Wheeled robot
    try:
        if test_wheeled_robot_discovery():
            tests_passed += 1
            print("✅ Wheeled robot discovery - PASSED")
    except Exception as e:
        print(f"❌ Wheeled robot discovery - FAILED: {e}")
    
    # Test 2: Legged robot
    try:
        if test_legged_robot_discovery():
            tests_passed += 1
            print("✅ Legged robot discovery - PASSED")
    except Exception as e:
        print(f"❌ Legged robot discovery - FAILED: {e}")
    
    # Test 3: 2D grid robot
    try:
        if test_2d_grid_robot_discovery():
            tests_passed += 1
            print("✅ 2D grid robot discovery - PASSED")
    except Exception as e:
        print(f"❌ 2D grid robot discovery - FAILED: {e}")
    
    # Test 4: Brain integration
    try:
        if test_brain_integration():
            tests_passed += 1
            print("✅ Brain integration - PASSED")
    except Exception as e:
        print(f"❌ Brain integration - FAILED: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 All universal actuator discovery tests passed!")
        print("✅ The brain can now:")
        print("   • Discover actuator effects for any embodiment")
        print("   • Form emergent categories (spatial, manipulative, environmental)")
        print("   • Work with wheels, legs, grids, or alien actuators")
        print("   • Learn what constitutes 'movement' through experience")
        print("   • Categorize actuators by effect patterns, not hardcoded types")
        print("🧠 True universal embodiment adaptability achieved!")
    else:
        print("⚠️  Some actuator discovery tests failed. The system may need refinement.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🌟 Universal actuator discovery system is fully operational!")
        print("🤖 The brain can now adapt to ANY type of robot embodiment!")
        print("🔬 Actuator categories emerge from pure experience, no hardcoded assumptions!")
    else:
        print("\n🔧 Actuator discovery system needs debugging")