#!/usr/bin/env python3
"""
Comprehensive Behavioral Test for UnifiedFieldBrain

Tests core robot behaviors to ensure optimizations haven't broken functionality:
1. Obstacle avoidance
2. Navigation and turning
3. Learning and adaptation
4. Memory formation
5. Behavioral consistency
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../server/src'))

import time
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path

from brains.field.core_brain import UnifiedFieldBrain

@dataclass
class BehaviorMetrics:
    """Metrics for behavioral analysis."""
    test_name: str
    cycles: int
    avg_obstacle_response: float
    avg_turn_response: float
    memory_formations: int
    behavioral_consistency: float
    avg_cycle_time: float
    success_rate: float
    
class SimulatedEnvironment:
    """Simple 2D environment for testing robot behaviors."""
    
    def __init__(self, size: float = 10.0):
        self.size = size
        self.robot_x = size / 2
        self.robot_y = size / 2
        self.robot_heading = 0.0
        
        # Fixed obstacles for consistent testing
        self.obstacles = [
            (2.0, 2.0, 1.0),  # (x, y, radius)
            (8.0, 8.0, 1.0),
            (2.0, 8.0, 0.8),
            (8.0, 2.0, 0.8),
        ]
        
    def get_sensor_readings(self) -> List[float]:
        """Get simulated sensor readings based on robot position."""
        # 3 front distance sensors + 21 other sensors
        sensors = []
        
        # Front sensors at -30°, 0°, +30°
        for angle_offset in [-np.pi/6, 0, np.pi/6]:
            angle = self.robot_heading + angle_offset
            distance = self._cast_ray(angle)
            # Convert to sensor reading (closer = higher value)
            sensors.append(max(0, 1.0 - distance / 5.0))
        
        # Fill remaining sensors with noise
        sensors.extend(np.random.rand(21) * 0.1)
        
        return sensors
    
    def _cast_ray(self, angle: float) -> float:
        """Cast a ray and find distance to nearest obstacle."""
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        min_dist = 5.0  # Max sensor range
        
        # Check walls
        if dx > 0:
            wall_dist = (self.size - self.robot_x) / dx
        elif dx < 0:
            wall_dist = -self.robot_x / dx
        else:
            wall_dist = float('inf')
        min_dist = min(min_dist, wall_dist)
        
        if dy > 0:
            wall_dist = (self.size - self.robot_y) / dy
        elif dy < 0:
            wall_dist = -self.robot_y / dy
        else:
            wall_dist = float('inf')
        min_dist = min(min_dist, wall_dist)
        
        # Check obstacles
        for ox, oy, radius in self.obstacles:
            # Simplified ray-circle intersection
            to_center = np.sqrt((ox - self.robot_x)**2 + (oy - self.robot_y)**2)
            if to_center > 0:
                angle_to_obstacle = np.arctan2(oy - self.robot_y, ox - self.robot_x)
                angle_diff = abs((angle - angle_to_obstacle + np.pi) % (2*np.pi) - np.pi)
                if angle_diff < np.pi/4:  # Within sensor cone
                    dist = max(0, to_center - radius)
                    min_dist = min(min_dist, dist)
        
        return min_dist
    
    def apply_action(self, action: List[float]):
        """Apply motor action to robot."""
        left_motor, right_motor, _, speed = action
        
        # Differential drive kinematics
        linear_vel = speed * 0.1  # Scale down
        angular_vel = (right_motor - left_motor) * 0.5
        
        # Update position
        self.robot_x += linear_vel * np.cos(self.robot_heading)
        self.robot_y += linear_vel * np.sin(self.robot_heading)
        self.robot_heading += angular_vel
        
        # Keep in bounds
        self.robot_x = np.clip(self.robot_x, 0, self.size)
        self.robot_y = np.clip(self.robot_y, 0, self.size)

def test_obstacle_avoidance(brain: UnifiedFieldBrain, cycles: int = 50) -> Dict:
    """Test obstacle avoidance behavior."""
    print("\n1. Testing Obstacle Avoidance:")
    
    responses = []
    
    for i in range(cycles):
        # Simulate obstacle at different distances
        distance = 1.0 - (i / cycles) * 0.9  # From far to very close
        sensors = [distance, distance, distance * 0.5] + [0.0] * 21
        
        action, _ = brain.process_robot_cycle(sensors)
        
        # Measure response (negative speed = backing up)
        obstacle_response = -action[3] if distance < 0.3 else 0
        responses.append(obstacle_response)
        
        if i % 10 == 0:
            print(f"   Distance: {distance:.2f}, Speed: {action[3]:.3f}")
    
    avg_response = np.mean(responses)
    success = avg_response > 0.1  # Should slow/reverse when close
    
    print(f"   Average obstacle response: {avg_response:.3f}")
    print(f"   {'✅' if success else '❌'} Obstacle avoidance {'working' if success else 'needs work'}")
    
    return {'avg_response': avg_response, 'success': success}

def test_turning_behavior(brain: UnifiedFieldBrain, cycles: int = 50) -> Dict:
    """Test turning behavior."""
    print("\n2. Testing Turning Behavior:")
    
    turn_responses = []
    
    # Test turning left (obstacle on right)
    for i in range(cycles // 2):
        sensors = [0.1, 0.5, 0.9] + [0.0] * 21  # Obstacle on right
        action, _ = brain.process_robot_cycle(sensors)
        turn = action[0] - action[1]  # Positive = turn left
        turn_responses.append(abs(turn))
    
    # Test turning right (obstacle on left)
    for i in range(cycles // 2):
        sensors = [0.9, 0.5, 0.1] + [0.0] * 21  # Obstacle on left
        action, _ = brain.process_robot_cycle(sensors)
        turn = action[1] - action[0]  # Positive = turn right
        turn_responses.append(abs(turn))
    
    avg_turn = np.mean(turn_responses)
    success = avg_turn > 0.05
    
    print(f"   Average turn response: {avg_turn:.3f}")
    print(f"   {'✅' if success else '❌'} Turning behavior {'working' if success else 'needs work'}")
    
    return {'avg_turn': avg_turn, 'success': success}

def test_behavioral_consistency(brain: UnifiedFieldBrain) -> Dict:
    """Test if behavior is consistent for same inputs."""
    print("\n3. Testing Behavioral Consistency:")
    
    test_input = [0.5, 0.7, 0.3] + [0.1] * 21
    actions = []
    
    for i in range(10):
        action, _ = brain.process_robot_cycle(test_input)
        actions.append(action)
    
    # Calculate variance in actions
    actions_array = np.array(actions)
    variances = np.var(actions_array, axis=0)
    avg_variance = np.mean(variances)
    
    # Lower variance = more consistent
    consistency = 1.0 - min(1.0, avg_variance * 10)
    success = consistency > 0.7
    
    print(f"   Behavioral consistency: {consistency:.3f}")
    print(f"   {'✅' if success else '❌'} Behavior is {'consistent' if success else 'inconsistent'}")
    
    return {'consistency': consistency, 'success': success}

def test_memory_formation(brain: UnifiedFieldBrain) -> Dict:
    """Test memory formation."""
    print("\n4. Testing Memory Formation:")
    
    initial_regions = len(brain.topology_regions)
    
    # Create distinctive patterns
    patterns = [
        np.sin(np.arange(24) * 0.5),
        np.cos(np.arange(24) * 0.3),
        np.sin(np.arange(24) * 0.7) * np.cos(np.arange(24) * 0.2)
    ]
    
    for pattern in patterns:
        for _ in range(20):  # Repeat each pattern
            brain.process_robot_cycle(list(pattern))
    
    final_regions = len(brain.topology_regions)
    new_memories = final_regions - initial_regions
    
    success = new_memories > 0
    
    print(f"   Initial regions: {initial_regions}")
    print(f"   Final regions: {final_regions}")
    print(f"   New memories formed: {new_memories}")
    print(f"   {'✅' if success else '❌'} Memory formation {'working' if success else 'needs work'}")
    
    return {'new_memories': new_memories, 'success': success}

def test_integrated_behavior(brain: UnifiedFieldBrain) -> Dict:
    """Test integrated behavior in simulated environment."""
    print("\n5. Testing Integrated Navigation:")
    
    env = SimulatedEnvironment()
    positions = []
    collisions = 0
    
    for cycle in range(100):
        sensors = env.get_sensor_readings()
        action, _ = brain.process_robot_cycle(sensors)
        
        old_pos = (env.robot_x, env.robot_y)
        env.apply_action(action)
        new_pos = (env.robot_x, env.robot_y)
        
        positions.append(new_pos)
        
        # Check for collision
        for ox, oy, radius in env.obstacles:
            dist = np.sqrt((env.robot_x - ox)**2 + (env.robot_y - oy)**2)
            if dist < radius + 0.5:
                collisions += 1
                break
        
        if cycle % 25 == 0:
            print(f"   Cycle {cycle}: Position ({env.robot_x:.1f}, {env.robot_y:.1f})")
    
    # Calculate total distance traveled
    total_distance = sum(
        np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        for p1, p2 in zip(positions[:-1], positions[1:])
    )
    
    success = total_distance > 5.0 and collisions < 10
    
    print(f"   Total distance traveled: {total_distance:.1f}")
    print(f"   Collisions: {collisions}")
    print(f"   {'✅' if success else '❌'} Navigation {'working' if success else 'needs improvement'}")
    
    return {'distance': total_distance, 'collisions': collisions, 'success': success}

def run_behavioral_tests(resolution: int = 3):
    """Run all behavioral tests."""
    print(f"\n{'='*60}")
    print(f"UNIFIED FIELD BRAIN BEHAVIORAL TESTS (Resolution {resolution}³)")
    print(f"{'='*60}")
    
    # Create brain
    brain = UnifiedFieldBrain(spatial_resolution=resolution, quiet_mode=True)
    
    # Track timing
    start_time = time.time()
    cycle_times = []
    
    # Run tests
    results = {}
    results['obstacle_avoidance'] = test_obstacle_avoidance(brain)
    results['turning'] = test_turning_behavior(brain)
    results['consistency'] = test_behavioral_consistency(brain)
    results['memory'] = test_memory_formation(brain)
    results['navigation'] = test_integrated_behavior(brain)
    
    # Performance metrics
    print("\n6. Performance Metrics:")
    for _ in range(10):
        start = time.perf_counter()
        brain.process_robot_cycle([0.5] * 24)
        cycle_times.append((time.perf_counter() - start) * 1000)
    
    avg_cycle_time = np.mean(cycle_times)
    print(f"   Average cycle time: {avg_cycle_time:.1f}ms")
    print(f"   Frequency: {1000/avg_cycle_time:.1f} Hz")
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.0f}%")
    
    # Save results
    output_dir = Path("logs/behavioral_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = BehaviorMetrics(
        test_name=f"unified_field_res{resolution}",
        cycles=brain.field_evolution_cycles,
        avg_obstacle_response=results['obstacle_avoidance']['avg_response'],
        avg_turn_response=results['turning']['avg_turn'],
        memory_formations=results['memory']['new_memories'],
        behavioral_consistency=results['consistency']['consistency'],
        avg_cycle_time=avg_cycle_time,
        success_rate=passed_tests/total_tests
    )
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"behavioral_test_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    brain.shutdown()
    return metrics

if __name__ == "__main__":
    # Run tests at different resolutions
    for res in [3, 5]:
        metrics = run_behavioral_tests(resolution=res)
        time.sleep(1)  # Brief pause between tests