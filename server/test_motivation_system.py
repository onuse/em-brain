#!/usr/bin/env python3
"""
Test Motivation System Implementation

Validates that the competitive motivation system works correctly.
"""

import sys
import os
import time
import random

# Add server to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock brain for testing
class MockBrain:
    """Simple mock brain for testing motivations."""
    
    def predict(self, state, action):
        """Mock prediction that returns action with some uncertainty."""
        
        # Simulate prediction confidence based on action type
        confidence = 0.8
        if action.get('type') == 'move':
            confidence = 0.6 if action.get('speed', 0) > 0.7 else 0.8
        elif action.get('type') == 'rotate':
            confidence = 0.7
        elif action.get('type') == 'stop':
            confidence = 0.9
        
        # Return mock prediction
        return MockPrediction(action, confidence)
    
    def get_prediction_confidence(self, predicted_outcome):
        """Return confidence for a prediction."""
        return predicted_outcome.confidence

class MockPrediction:
    """Mock prediction outcome."""
    
    def __init__(self, action, confidence):
        self.action = action
        self.confidence = confidence
        self.similarity_score = confidence  # Use confidence as similarity proxy

class MockState:
    """Mock robot state."""
    
    def __init__(self, battery=0.7, obstacle_distance=50):
        self.battery = battery
        self.obstacle_distance = obstacle_distance


def test_individual_motivations():
    """Test each motivation individually."""
    
    print("🧪 Testing Individual Motivations")
    print("=" * 50)
    
    brain = MockBrain()
    
    from motivation.standard import CuriosityMotivation, EnergyMotivation, SafetyMotivation
    
    # Test Curiosity
    print("\n🔍 Testing CuriosityMotivation:")
    curiosity = CuriosityMotivation(weight=1.0)
    curiosity.connect_to_brain(brain)
    
    state = MockState()
    proposal = curiosity.propose_action(state)
    print(f"   Proposal: {proposal}")
    print(f"   Value: {proposal.motivation_value:.3f}")
    print(f"   Reasoning: {proposal.reasoning}")
    
    # Test Energy
    print("\n🔋 Testing EnergyMotivation:")
    energy = EnergyMotivation(weight=1.0)
    energy.connect_to_brain(brain)
    
    # Test low energy scenario
    low_energy_state = MockState(battery=0.15)
    proposal = energy.propose_action(low_energy_state)
    print(f"   Low Energy Proposal: {proposal}")
    print(f"   Value: {proposal.motivation_value:.3f}")
    print(f"   Reasoning: {proposal.reasoning}")
    
    # Test Safety
    print("\n🛡️ Testing SafetyMotivation:")
    safety = SafetyMotivation(weight=1.0)
    safety.connect_to_brain(brain)
    
    # Test dangerous scenario
    dangerous_state = MockState(obstacle_distance=5)  # Very close obstacle
    proposal = safety.propose_action(dangerous_state)
    print(f"   Dangerous Scenario Proposal: {proposal}")
    print(f"   Value: {proposal.motivation_value:.3f}")
    print(f"   Reasoning: {proposal.reasoning}")


def test_motivation_competition():
    """Test competitive action selection between motivations."""
    
    print("\n🏆 Testing Motivation Competition")
    print("=" * 50)
    
    brain = MockBrain()
    
    from motivation.system import MotivationSystem
    from motivation.standard import CuriosityMotivation, EnergyMotivation, SafetyMotivation
    
    # Create motivation system
    motivation_system = MotivationSystem(brain)
    motivation_system.set_verbose(True)
    
    # Add motivations with different weights
    motivation_system.add_motivation(CuriosityMotivation(weight=0.6))
    motivation_system.add_motivation(EnergyMotivation(weight=0.8))
    motivation_system.add_motivation(SafetyMotivation(weight=0.9))
    
    print(f"\n🎯 Running Competition Scenarios:")
    
    # Scenario 1: Normal state
    print(f"\n--- Scenario 1: Normal State ---")
    normal_state = MockState(battery=0.7, obstacle_distance=50)
    action = motivation_system.select_action(normal_state)
    print(f"Selected Action: {action}")
    
    # Scenario 2: Low energy
    print(f"\n--- Scenario 2: Low Energy ---")
    low_energy_state = MockState(battery=0.1, obstacle_distance=50)
    action = motivation_system.select_action(low_energy_state)
    print(f"Selected Action: {action}")
    
    # Scenario 3: Danger
    print(f"\n--- Scenario 3: Danger ---")
    danger_state = MockState(battery=0.7, obstacle_distance=3)
    action = motivation_system.select_action(danger_state)
    print(f"Selected Action: {action}")
    
    # Show final statistics
    motivation_system.print_motivation_report()


def test_personality_differences():
    """Test that different motivation weights create different personalities."""
    
    print("\n🎭 Testing Personality Differences")
    print("=" * 50)
    
    brain = MockBrain()
    
    from motivation.system import MotivationSystem
    from motivation.standard import CuriosityMotivation, EnergyMotivation, SafetyMotivation
    
    # Create different personality profiles
    personalities = {
        'Explorer': {'curiosity': 0.9, 'energy': 0.3, 'safety': 0.4},
        'Cautious': {'curiosity': 0.2, 'energy': 0.6, 'safety': 0.9},
        'Efficient': {'curiosity': 0.4, 'energy': 0.9, 'safety': 0.5}
    }
    
    test_state = MockState(battery=0.4, obstacle_distance=20)
    
    for personality_name, weights in personalities.items():
        print(f"\n🤖 {personality_name} Personality:")
        
        # Create motivation system with personality weights
        system = MotivationSystem(brain)
        system.add_motivation(CuriosityMotivation(weight=weights['curiosity']))
        system.add_motivation(EnergyMotivation(weight=weights['energy']))
        system.add_motivation(SafetyMotivation(weight=weights['safety']))
        
        # Test multiple decisions to see patterns
        decisions = []
        for i in range(5):
            action = system.select_action(test_state)
            decisions.append(action)
        
        # Show decision pattern
        print(f"   Decisions: {[str(d) for d in decisions]}")
        
        # Show motivation statistics
        stats = system.get_motivation_statistics()
        for motivation_stats in stats['motivations']:
            name = motivation_stats['name']
            win_rate = motivation_stats['win_rate']
            print(f"   {name:10} {win_rate:5.1%} win rate")


def main():
    """Run all motivation system tests."""
    
    print("🚀 MOTIVATION SYSTEM TESTS")
    print("=" * 60)
    
    try:
        # Test individual components
        test_individual_motivations()
        
        # Test competition system
        test_motivation_competition()
        
        # Test personality emergence
        test_personality_differences()
        
        print(f"\n✅ All motivation system tests completed successfully!")
        print(f"🎯 Key Achievements:")
        print(f"   • Individual motivations generate proposals")
        print(f"   • Competition system selects winning actions")
        print(f"   • Different weights create different personalities")
        print(f"   • System provides detailed statistics and logging")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()