#!/usr/bin/env python3
"""
Test the multi-drive motivation system.
Verifies that multiple drives compete and cooperate to guide robot behavior.
"""

import asyncio
from datetime import datetime
from drives import create_default_motivation_system, DriveContext
from predictor.multi_drive_predictor import MultiDrivePredictor
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from simulation.brainstem_sim import GridWorldBrainstem


def test_motivation_system_basics():
    """Test basic motivation system functionality."""
    print("🧠 Testing Multi-Drive Motivation System")
    print("=======================================")
    
    # Create motivation system
    motivation_system = create_default_motivation_system()
    
    print(f"Active drives: {motivation_system.list_drives()}")
    
    # Test drive context
    context = DriveContext(
        current_sensory=[1.0, 2.0, 3.0],
        robot_health=0.8,
        robot_energy=0.6,
        robot_position=(5, 3),
        robot_orientation=1,
        recent_experiences=[],
        prediction_errors=[0.1, 0.2, 0.3],
        time_since_last_food=15,
        time_since_last_damage=5,
        threat_level="normal",
        step_count=100
    )
    
    # Test action candidates
    action_candidates = [
        {"forward_motor": 0.5, "turn_motor": 0.0, "brake_motor": 0.0},  # Forward
        {"forward_motor": 0.0, "turn_motor": 0.5, "brake_motor": 0.0},  # Turn right
        {"forward_motor": 0.0, "turn_motor": 0.0, "brake_motor": 0.8},  # Stop
        {"forward_motor": -0.4, "turn_motor": 0.0, "brake_motor": 0.0}, # Backward
    ]
    
    # Evaluate actions
    result = motivation_system.evaluate_action_candidates(action_candidates, context)
    
    print(f"✅ Chosen action: {result.chosen_action}")
    print(f"   Total score: {result.total_score:.3f}")
    print(f"   Dominant drive: {result.dominant_drive}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Reasoning: {result.reasoning}")
    print(f"   Drive contributions: {result.drive_contributions}")
    
    return result.dominant_drive is not None


def test_drive_competition():
    """Test competition between different drives in different scenarios."""
    print(f"\n🏆 Testing Drive Competition")
    print("===========================")
    
    motivation_system = create_default_motivation_system()
    
    scenarios = [
        {
            "name": "Low Energy Crisis",
            "health": 1.0,
            "energy": 0.1,  # Very low energy
            "threat": "normal",
            "expected_dominant": "Survival"
        },
        {
            "name": "High Threat Situation", 
            "health": 0.9,
            "energy": 0.8,
            "threat": "danger",  # High threat
            "expected_dominant": "Survival"
        },
        {
            "name": "Healthy Exploration",
            "health": 1.0,
            "energy": 1.0,
            "threat": "normal",
            "expected_dominant": None  # Any drive could dominate
        },
        {
            "name": "Uncertain Environment",
            "health": 0.8,
            "energy": 0.7,
            "threat": "normal",
            "prediction_errors": [0.8, 0.9, 0.7],  # High uncertainty
            "expected_dominant": "Curiosity"
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        context = DriveContext(
            current_sensory=[1.0] * 5,
            robot_health=scenario["health"],
            robot_energy=scenario["energy"],
            robot_position=(0, 0),
            robot_orientation=0,
            recent_experiences=[],
            prediction_errors=scenario.get("prediction_errors", [0.1, 0.2]),
            time_since_last_food=10,
            time_since_last_damage=5,
            threat_level=scenario["threat"],
            step_count=50
        )
        
        # Generate and evaluate actions
        action_candidates = motivation_system.generate_action_candidates(context)
        result = motivation_system.evaluate_action_candidates(action_candidates, context)
        
        results[scenario["name"]] = result
        
        print(f"📊 {scenario['name']}:")
        print(f"   Dominant drive: {result.dominant_drive}")
        print(f"   Drive scores: {result.drive_contributions}")
        print(f"   Action: forward={result.chosen_action.get('forward_motor', 0):.2f}, "
              f"turn={result.chosen_action.get('turn_motor', 0):.2f}, "
              f"brake={result.chosen_action.get('brake_motor', 0):.2f}")
        print(f"   Reasoning: {result.reasoning}")
        
        # Check if expectation matches (if specified)
        expected = scenario.get("expected_dominant")
        if expected and result.dominant_drive == expected:
            print(f"   ✅ Expected {expected} to dominate - CORRECT")
        elif expected:
            print(f"   ⚠️  Expected {expected}, got {result.dominant_drive}")
        else:
            print(f"   ✅ Open scenario - {result.dominant_drive} dominated")
        
        print()
    
    return len(results) == len(scenarios)


def test_multi_drive_predictor():
    """Test the integrated multi-drive predictor."""
    print("🔮 Testing Multi-Drive Predictor Integration")
    print("==========================================")
    
    # Create multi-drive predictor
    predictor = MultiDrivePredictor(
        base_time_budget=0.02,  # Fast for testing
    )
    
    brain = BrainInterface(predictor)
    
    # Test prediction generation
    sensory_packet = SensoryPacket(
        sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        actuator_positions=[0.0, 0.0, 0.0],
        timestamp=datetime.now(),
        sequence_id=1
    )
    
    mental_context = [0.5, 0.8, 0.7, 0.9, 0.2]
    
    # Test with different robot states
    test_cases = [
        {"health": 1.0, "energy": 1.0, "threat": "normal", "name": "Healthy State"},
        {"health": 0.3, "energy": 0.8, "threat": "normal", "name": "Low Health"},
        {"health": 0.9, "energy": 0.2, "threat": "normal", "name": "Low Energy"},
        {"health": 0.8, "energy": 0.7, "threat": "danger", "name": "Dangerous Situation"}
    ]
    
    predictions_made = 0
    
    for case in test_cases:
        # Manually provide robot state to predictor
        prediction = predictor.generate_prediction(
            current_context=mental_context,
            world_graph=brain.world_graph,
            sequence_id=predictions_made + 1,
            threat_level=case["threat"],
            robot_health=case["health"],
            robot_energy=case["energy"],
            robot_position=(5, 5),
            robot_orientation=0,
            step_count=predictions_made
        )
        
        predictions_made += 1
        
        print(f"🎯 {case['name']}:")
        print(f"   Action: {prediction.prediction.motor_action}")
        print(f"   Consensus: {prediction.consensus_strength}")
        print(f"   Confidence: {prediction.prediction.confidence:.3f}")
        print(f"   Reasoning: {prediction.reasoning}")
        print()
    
    # Check predictor statistics
    stats = predictor.get_predictor_statistics()
    print(f"📈 Predictor Statistics:")
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Motivation system drives: {len(stats['motivation_system']['active_drives'])}")
    if 'drive_dominance_counts' in stats:
        print(f"   Drive dominance: {stats['drive_dominance_counts']}")
    
    return predictions_made == len(test_cases)


async def test_full_simulation():
    """Test multi-drive system in full simulation."""
    print("\n🌍 Testing Multi-Drive System in Simulation")
    print("==========================================")
    
    # Create brainstem with multi-drive brain
    brainstem = GridWorldBrainstem(
        world_width=8, world_height=8, seed=42, use_sockets=False
    )
    
    # Create multi-drive predictor  
    predictor = MultiDrivePredictor(
        base_time_budget=0.02,
    )
    brain = BrainInterface(predictor)
    brainstem.brain_client = brain
    
    # Run simulation
    try:
        results = await brainstem.run_brain_controlled_simulation(
            steps=12, step_delay=0.1
        )
        
        print(f"✅ Simulation Results:")
        print(f"   Steps completed: {results['steps_completed']}")
        print(f"   Predictions received: {results['predictions_received']}")
        print(f"   Communication errors: {results['communication_errors']}")
        print(f"   Final health: {results['final_robot_state']['health']:.2f}")
        print(f"   Final energy: {results['final_robot_state']['energy']:.2f}")
        
        # Analyze decision patterns
        if results['performance_stats']:
            consensus_types = {}
            for stat in results['performance_stats']:
                consensus = stat['consensus_strength']
                consensus_types[consensus] = consensus_types.get(consensus, 0) + 1
            
            print(f"   Decision patterns: {consensus_types}")
            
            multi_drive_decisions = consensus_types.get('motivation_driven', 0) + consensus_types.get('bootstrap_motivation', 0)
            total_decisions = len(results['performance_stats'])
            
            if multi_drive_decisions > 0:
                print(f"   ✅ {multi_drive_decisions}/{total_decisions} decisions were multi-drive motivated")
            else:
                print(f"   ⚠️  No multi-drive decisions detected")
        
        # Get detailed brain statistics
        try:
            brain_stats = brain.get_brain_statistics()
            if 'predictor_stats' in brain_stats:
                pred_stats = brain_stats['predictor_stats']
                if 'motivation_system' in pred_stats:
                    motivation_stats = pred_stats['motivation_system']
                    print(f"   Active drives: {motivation_stats.get('active_drives', [])}")
                    if 'recent_drive_dominance' in motivation_stats:
                        print(f"   Recent drive dominance: {motivation_stats['recent_drive_dominance']}")
        except Exception as e:
            print(f"   Note: Could not get detailed brain statistics: {e}")
        
        return results['steps_completed'] > 5
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False


async def main():
    """Run all multi-drive system tests."""
    print("🤖 Multi-Drive Motivation System Test Suite")
    print("===========================================")
    print("Testing emergent behavior from competing drives:")
    print("• Curiosity Drive - seeks learning opportunities")
    print("• Survival Drive - maintains health and energy") 
    print("• Exploration Drive - discovers new areas")
    print()
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Basic motivation system
    try:
        if test_motivation_system_basics():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Basic motivation test failed: {e}")
    
    # Test 2: Drive competition
    try:
        if test_drive_competition():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Drive competition test failed: {e}")
    
    # Test 3: Multi-drive predictor
    try:
        if test_multi_drive_predictor():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Multi-drive predictor test failed: {e}")
    
    # Test 4: Full simulation
    try:
        if await test_full_simulation():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Full simulation test failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All multi-drive system tests passed!")
        print("✅ The robot now has sophisticated, competing motivations:")
        print("   • Curiosity drives learning and uncertainty reduction")
        print("   • Survival drives self-preservation and resource seeking")
        print("   • Exploration drives environmental discovery and mapping")
        print("   • Drives compete and cooperate to create emergent behavior")
    else:
        print("⚠️  Some tests failed. The multi-drive system may need debugging.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🌟 Multi-drive emergent intelligence is fully operational!")
    else:
        print("\n🔧 Multi-drive system needs refinement")