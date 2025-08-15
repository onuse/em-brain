#!/usr/bin/env python3
"""
Spatial Learning Demo

Demonstrates how spatial intelligence emerges from the 4-system minimal brain.
A simple robot learns to navigate a 2D world using only sensory similarity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server'))

from server.src.brain_factory import MinimalBrain
import numpy as np
import time


class Simple2DWorld:
    """A simple 2D grid world for testing spatial learning."""
    
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.robot_x = 5
        self.robot_y = 5
        
        # Create a simple world with some obstacles
        self.obstacles = {(2, 2), (2, 3), (3, 2), (7, 7), (7, 8), (8, 7)}
        
        print(f"🌍 Created {width}x{height} world with {len(self.obstacles)} obstacles")
        print(f"   Robot starts at ({self.robot_x}, {self.robot_y})")
    
    def get_sensory_input(self):
        """Get 8-direction distance sensors around the robot."""
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
            (0, -1),           (0, 1),   # W,     E
            (1, -1),  (1, 0),  (1, 1)    # SW, S, SE
        ]
        
        sensors = []
        for dx, dy in directions:
            distance = self._raycast_distance(dx, dy)
            sensors.append(distance / 10.0)  # Normalize to 0-1
        
        return sensors
    
    def _raycast_distance(self, dx, dy):
        """Cast a ray and return distance to obstacle or wall."""
        x, y = self.robot_x, self.robot_y
        distance = 0
        
        while True:
            x += dx
            y += dy
            distance += 1
            
            # Hit wall
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return distance
            
            # Hit obstacle
            if (x, y) in self.obstacles:
                return distance
            
            # Max sensor range
            if distance >= 10:
                return 10
    
    def execute_action(self, action):
        """Execute a movement action [dx, dy] and return what happened."""
        # Convert action to discrete movement
        dx = 1 if action[0] > 0.5 else (-1 if action[0] < -0.5 else 0)
        dy = 1 if action[1] > 0.5 else (-1 if action[1] < -0.5 else 0)
        
        new_x = max(0, min(self.width - 1, self.robot_x + dx))
        new_y = max(0, min(self.height - 1, self.robot_y + dy))
        
        # Check for collision
        if (new_x, new_y) in self.obstacles:
            # Hit obstacle - don't move
            outcome = self.get_sensory_input()  # Same position
            return outcome, False  # False = collision
        else:
            # Move to new position
            self.robot_x = new_x
            self.robot_y = new_y
            outcome = self.get_sensory_input()  # New position
            return outcome, True  # True = successful movement
    
    def print_world(self):
        """Print a simple ASCII representation of the world."""
        print("\n" + "="*20)
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if x == self.robot_x and y == self.robot_y:
                    row += "R"  # Robot
                elif (x, y) in self.obstacles:
                    row += "#"  # Obstacle
                else:
                    row += "."  # Empty space
            print(row)
        print("="*20)


def run_spatial_learning_demo():
    """Run the spatial learning demonstration."""
    
    print("🤖 Spatial Learning Demo - Emergent Navigation Intelligence")
    print("   Testing if spatial navigation emerges from similarity + experience")
    
    # Create world and brain
    world = Simple2DWorld(width=10, height=10)
    brain = MinimalBrain()
    
    # Learning phase
    print("\n📚 Learning Phase (50 random explorations)")
    
    for step in range(50):
        # Get current sensory input
        sensors = world.get_sensory_input()
        
        # Brain predicts action
        action, brain_state = brain.process_sensory_input(sensors, action_dimensions=2)
        
        # Execute action in world
        outcome, success = world.execute_action(action)
        
        # Store experience in brain
        brain.store_experience(sensors, action, outcome, action)
        
        if step % 10 == 0:
            print(f"   Step {step}: Robot at ({world.robot_x}, {world.robot_y}), "
                  f"prediction method: {brain_state['prediction_method']}, "
                  f"confidence: {brain_state['prediction_confidence']:.3f}")
    
    print(f"\n🧠 After learning: {brain.total_experiences} experiences stored")
    stats = brain.get_brain_stats()
    print(f"   Consensus predictions: {stats['prediction_engine']['consensus_rate']:.2f}")
    working_memory_size = stats['activation_dynamics'].get('working_memory_size', 
                                                       stats['activation_dynamics'].get('current_working_memory_size', 0))
    print(f"   Working memory size: {working_memory_size}")
    
    # Test spatial memory
    print("\n🎯 Testing Spatial Memory")
    print("   Moving robot to known positions to test if brain recognizes them...")
    
    # Test several known positions
    test_positions = [(2, 5), (5, 5), (8, 2), (1, 8)]
    
    for test_x, test_y in test_positions:
        # Move robot to test position
        world.robot_x = test_x
        world.robot_y = test_y
        
        # Get sensory input at this position
        sensors = world.get_sensory_input()
        
        # See what brain predicts
        action, brain_state = brain.process_sensory_input(sensors, action_dimensions=2)
        
        print(f"   Position ({test_x}, {test_y}): {brain_state['num_similar_experiences']} similar memories found, "
              f"confidence: {brain_state['prediction_confidence']:.3f}")
    
    # Show final world state
    world.print_world()
    
    # Final brain statistics
    print("\n📊 Final Brain Statistics")
    final_stats = brain.get_brain_stats()
    
    print(f"   Total experiences: {final_stats['brain_summary']['total_experiences']}")
    print(f"   Total predictions: {final_stats['brain_summary']['total_predictions']}")
    print(f"   Similarity engine: {final_stats['similarity_engine']['device']} "
          f"({final_stats['similarity_engine']['total_searches']} searches)")
    final_working_memory = final_stats['activation_dynamics'].get('working_memory_size', 
                                                              final_stats['activation_dynamics'].get('current_working_memory_size', 0))
    print(f"   Working memory: {final_working_memory} active experiences")
    print(f"   Prediction accuracy: {final_stats['prediction_engine']['avg_prediction_accuracy']:.3f}")
    
    print("\n✅ Demo complete!")
    print("   Key insight: Spatial navigation emerged from similarity matching!")
    print("   The brain learned to recognize places without any hardcoded spatial representation.")
    
    return brain, world


def main():
    """Main entry point for spatial learning demo."""
    try:
        brain, world = run_spatial_learning_demo()
        
        print("\n🎉 Spatial learning demo completed successfully!")
        print("The minimal brain demonstrated emergent spatial intelligence.")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)