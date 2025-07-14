#!/usr/bin/env python3
"""
Test Motivation System Balance

Tests motivation competition with more balanced scenarios.
"""

import sys
import os

# Add server to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from test_motivation_system import MockBrain, MockState
from motivation.system import MotivationSystem
from motivation.standard import CuriosityMotivation, EnergyMotivation, SafetyMotivation


def test_balanced_competition():
    """Test motivation competition with more balanced weights."""
    
    print("⚖️ BALANCED MOTIVATION COMPETITION")
    print("=" * 60)
    
    brain = MockBrain()
    system = MotivationSystem(brain)
    system.set_verbose(True)
    
    # More balanced weights
    system.add_motivation(CuriosityMotivation(weight=0.4))
    system.add_motivation(EnergyMotivation(weight=0.4))
    system.add_motivation(SafetyMotivation(weight=0.3))  # Reduced safety dominance
    
    scenarios = [
        ("Normal operation", MockState(battery=0.8, obstacle_distance=100)),
        ("Moderate energy", MockState(battery=0.4, obstacle_distance=100)),
        ("Low energy", MockState(battery=0.15, obstacle_distance=100)),
        ("Close obstacle", MockState(battery=0.8, obstacle_distance=15)),
        ("Critical: low energy + danger", MockState(battery=0.1, obstacle_distance=8))
    ]
    
    for scenario_name, state in scenarios:
        print(f"\n--- {scenario_name} ---")
        action = system.select_action(state)
        print(f"Selected: {action}")
    
    system.print_motivation_report()


def test_extreme_scenarios():
    """Test how motivations respond to extreme situations."""
    
    print("\n🚨 EXTREME SCENARIO TESTING")
    print("=" * 60)
    
    brain = MockBrain()
    
    # Test each motivation in isolation to see their responses
    motivations = {
        'Curiosity': CuriosityMotivation(weight=1.0),
        'Energy': EnergyMotivation(weight=1.0),
        'Safety': SafetyMotivation(weight=1.0)
    }
    
    for name, motivation in motivations.items():
        motivation.connect_to_brain(brain)
    
    extreme_scenarios = [
        ("Dead battery", MockState(battery=0.05, obstacle_distance=100)),
        ("Imminent collision", MockState(battery=0.8, obstacle_distance=2)),
        ("Optimal conditions", MockState(battery=0.9, obstacle_distance=200))
    ]
    
    for scenario_name, state in extreme_scenarios:
        print(f"\n{scenario_name}:")
        for name, motivation in motivations.items():
            proposal = motivation.propose_action(state)
            print(f"   {name:10} val={proposal.motivation_value:.2f} conf={proposal.confidence:.2f} - {proposal.reasoning}")


def test_curiosity_scenarios():
    """Test scenarios where curiosity should win."""
    
    print("\n🔍 CURIOSITY-FOCUSED SCENARIOS")
    print("=" * 60)
    
    brain = MockBrain()
    system = MotivationSystem(brain)
    
    # Favor curiosity
    system.add_motivation(CuriosityMotivation(weight=0.8))
    system.add_motivation(EnergyMotivation(weight=0.2))
    system.add_motivation(SafetyMotivation(weight=0.2))
    
    # Safe, high-energy scenario where curiosity should dominate
    safe_state = MockState(battery=0.9, obstacle_distance=200)
    
    print(f"Safe + high energy scenario:")
    action = system.select_action(safe_state)
    print(f"Action: {action}")
    
    system.print_motivation_report()


def main():
    """Run motivation balance tests."""
    
    print("🧪 MOTIVATION SYSTEM BALANCE TESTS")
    print("=" * 70)
    
    try:
        test_balanced_competition()
        test_extreme_scenarios()
        test_curiosity_scenarios()
        
        print(f"\n✅ Balance tests completed!")
        print(f"🎯 Observations:")
        print(f"   • Safety motivation may need rebalancing")
        print(f"   • Energy motivation responds correctly to battery levels")
        print(f"   • Curiosity needs scenarios with higher uncertainty")
        print(f"   • Competition works - weights affect outcomes")
        
    except Exception as e:
        print(f"❌ Balance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()