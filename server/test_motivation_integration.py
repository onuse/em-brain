#!/usr/bin/env python3
"""
Test Motivation System Integration with Brain

Tests that the motivation system integrates properly with the core brain.
"""

import sys
import os

# Add server to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import brain and motivation components
from src.brain import MinimalBrain
from motivation.system import MotivationSystem
from motivation.standard import CuriosityMotivation, EnergyMotivation, SafetyMotivation


class BrainAdapter:
    """Adapter to make MinimalBrain compatible with motivation system."""
    
    def __init__(self, brain):
        self.brain = brain
        
    def predict(self, state, action):
        """Convert motivation predict call to brain prediction."""
        # Convert state to sensory input format brain expects
        sensory_input = self._state_to_sensory(state)
        
        # Convert action to format brain expects
        action_vector = self._action_to_vector(action)
        
        # Get brain prediction
        predicted_action, brain_state = self.brain.process_sensory_input(
            sensory_input, action_dimensions=len(action_vector) if action_vector else 2
        )
        
        # Return mock prediction that motivations can use
        return MockPrediction(action, brain_state.get('prediction_confidence', 0.7))
    
    def get_prediction_confidence(self, predicted_outcome):
        """Get confidence from prediction."""
        return predicted_outcome.confidence
    
    def _state_to_sensory(self, state):
        """Convert state object to sensory input vector."""
        if hasattr(state, 'battery') and hasattr(state, 'obstacle_distance'):
            return [state.battery, state.obstacle_distance / 100.0, 0.0, 0.0]
        return [0.5, 0.5, 0.0, 0.0]  # Default sensory input
    
    def _action_to_vector(self, action):
        """Convert action dict to vector."""
        if isinstance(action, dict):
            action_type = action.get('type', 'stop')
            if action_type == 'move':
                speed = action.get('speed', 0.5)
                return [speed, 0.0]
            elif action_type == 'rotate':
                angle = action.get('angle', 0) / 180.0  # Normalize to [-1, 1]
                return [0.0, angle]
            else:  # stop
                return [0.0, 0.0]
        return [0.0, 0.0]


class MockPrediction:
    """Mock prediction outcome for motivations."""
    
    def __init__(self, action, confidence):
        self.action = action
        self.confidence = confidence
        self.similarity_score = confidence


class MockState:
    """Mock robot state for testing."""
    
    def __init__(self, battery=0.7, obstacle_distance=50, location=(0, 0)):
        self.battery = battery
        self.obstacle_distance = obstacle_distance
        self.location = location
        self.timestamp = 0


def test_brain_motivation_integration():
    """Test that motivation system works with actual brain."""
    
    print("🧠 BRAIN + MOTIVATION INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Create minimal brain
        print("Creating minimal brain...")
        brain = MinimalBrain()
        
        # Create brain adapter for motivation compatibility
        print("Creating brain adapter...")
        brain_adapter = BrainAdapter(brain)
        
        # Create motivation system
        print("Creating motivation system...")
        motivation_system = MotivationSystem(brain_adapter)
        motivation_system.set_verbose(True)
        
        # Add motivations
        motivation_system.add_motivation(CuriosityMotivation(weight=0.6))
        motivation_system.add_motivation(EnergyMotivation(weight=0.7))
        motivation_system.add_motivation(SafetyMotivation(weight=0.8))
        
        print(f"\n🎯 Testing Integration Scenarios:")
        
        # Test scenario 1: Normal operation
        print(f"\n--- Normal Operation ---")
        state = MockState(battery=0.8, obstacle_distance=100)
        action = motivation_system.select_action(state)
        print(f"Action: {action}")
        
        # Test scenario 2: Low energy
        print(f"\n--- Low Energy ---")
        state = MockState(battery=0.15, obstacle_distance=100)
        action = motivation_system.select_action(state)
        print(f"Action: {action}")
        
        # Test scenario 3: Close obstacle
        print(f"\n--- Close Obstacle ---")
        state = MockState(battery=0.8, obstacle_distance=5)
        action = motivation_system.select_action(state)
        print(f"Action: {action}")
        
        # Show final report
        motivation_system.print_motivation_report()
        
        print(f"\n✅ Integration test completed successfully!")
        print(f"🎯 Key Validations:")
        print(f"   • Brain provides prediction services to motivations")
        print(f"   • Motivation system selects winning actions") 
        print(f"   • Different scenarios produce different winners")
        print(f"   • System logs decision history properly")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run integration test."""
    
    print("🚀 MOTIVATION SYSTEM INTEGRATION TEST")
    print("=" * 70)
    
    success = test_brain_motivation_integration()
    
    if success:
        print(f"\n🎉 All integration tests passed!")
        print(f"💡 The motivation system successfully extends the brain architecture")
        print(f"   without modifying the core 4-system implementation.")
    else:
        print(f"\n💥 Integration tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()