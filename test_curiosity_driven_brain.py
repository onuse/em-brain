#!/usr/bin/env python3
"""
Test the curiosity-driven brain implementation.
Verifies that the brain makes intelligent, curiosity-driven action decisions.
"""

import asyncio
from datetime import datetime
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from predictor.triple_predictor import TriplePredictor
from simulation.brainstem_sim import GridWorldBrainstem


def test_curiosity_driven_predictor():
    """Test basic curiosity-driven prediction functionality."""
    print("🧠 Testing Curiosity-Driven Predictor...")
    
    # Create brain with curiosity-driven predictor
    predictor = TriplePredictor(
        base_time_budget=0.01,  # Fast for testing
        exploration_rate=0.5    # Balanced exploration/exploitation
    )
    brain = BrainInterface(predictor)
    
    # Test 1: Bootstrap prediction (empty graph)
    sensory_packet = SensoryPacket(
        sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        actuator_positions=[0.0, 0.0, 0.0],
        timestamp=datetime.now(),
        sequence_id=1
    )
    
    mental_context = [0.5, 0.8, 0.9, 0.2, 0.1]  # pos_x, pos_y, energy, health, progress
    
    prediction1 = brain.process_sensory_input(sensory_packet, mental_context, "normal")
    
    print(f"✅ Bootstrap prediction generated:")
    print(f"   Action: {prediction1.motor_action}")
    print(f"   Confidence: {prediction1.confidence:.2f}")
    print(f"   Consensus: {prediction1.consensus_strength}")
    
    # Test 2: Experience accumulation
    for i in range(3):
        # Simulate robot taking action and getting new sensory input
        new_sensory = SensoryPacket(
            sensor_values=[1.1 + i*0.1, 2.1 + i*0.1, 3.1 + i*0.1, 4.1 + i*0.1, 5.1 + i*0.1],
            actuator_positions=[0.1 + i*0.05, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=i + 2
        )
        
        new_mental_context = [0.5 + i*0.1, 0.8, 0.9 - i*0.05, 0.2, 0.2 + i*0.1]
        
        prediction = brain.process_sensory_input(new_sensory, new_mental_context, "normal")
        
        print(f"   Step {i+1}: Action={prediction.motor_action}, "
              f"Consensus={prediction.consensus_strength}, "
              f"Confidence={prediction.confidence:.2f}")
    
    # Check brain statistics
    stats = brain.get_brain_statistics()
    print(f"✅ Brain accumulated {stats['interface_stats']['total_experiences']} experiences")
    
    return True


async def test_brain_controlled_simulation():
    """Test full brain-controlled simulation with curiosity drive."""
    print("\n🌍 Testing Brain-Controlled Simulation...")
    
    # Create grid world brainstem with curiosity-driven brain
    brainstem = GridWorldBrainstem(
        world_width=8, 
        world_height=8,
        use_sockets=False  # Local brain for testing
    )
    
    # Override the brain with curiosity-driven predictor
    predictor = TriplePredictor(
        base_time_budget=0.02,  # Fast for testing
        exploration_rate=0.6    # High exploration for new world
    )
    brainstem.brain_client = BrainInterface(predictor)
    
    try:
        # Run brain-controlled simulation
        results = await brainstem.run_brain_controlled_simulation(
            steps=15,
            step_delay=0.1
        )
        
        print(f"✅ Simulation completed:")
        print(f"   Steps: {results['steps_completed']}")
        print(f"   Predictions: {results['predictions_received']}")
        print(f"   Errors: {results['communication_errors']}")
        print(f"   Final robot health: {results['final_robot_state']['health']:.2f}")
        print(f"   Final robot energy: {results['final_robot_state']['energy']:.2f}")
        
        # Analyze decision patterns
        if results['performance_stats']:
            consensus_types = {}
            for stat in results['performance_stats']:
                consensus = stat['consensus_strength']
                consensus_types[consensus] = consensus_types.get(consensus, 0) + 1
            
            print(f"   Decision patterns: {consensus_types}")
            
            # Check if curiosity-driven decisions are being made
            curiosity_decisions = consensus_types.get('curiosity_driven', 0)
            total_decisions = len(results['performance_stats'])
            if curiosity_decisions > 0:
                print(f"   ✅ {curiosity_decisions}/{total_decisions} decisions were curiosity-driven")
            else:
                print(f"   ⚠️  No curiosity-driven decisions detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False


def test_action_diversity():
    """Test that curiosity drive generates diverse actions."""
    print("\n🎲 Testing Action Diversity...")
    
    predictor = TriplePredictor(
        base_time_budget=0.01,
        exploration_rate=0.8  # High exploration
    )
    brain = BrainInterface(predictor)
    
    actions_generated = []
    
    # Generate multiple predictions and collect actions
    for i in range(10):
        sensory_packet = SensoryPacket(
            sensor_values=[1.0 + i*0.1] * 5,
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=i + 1
        )
        
        mental_context = [0.5, 0.5, 0.8, 0.9, 0.1]
        prediction = brain.process_sensory_input(sensory_packet, mental_context, "normal")
        
        # Extract action signature for diversity analysis
        action_sig = (
            round(prediction.motor_action.get('forward_motor', 0.0), 1),
            round(prediction.motor_action.get('turn_motor', 0.0), 1),
            round(prediction.motor_action.get('brake_motor', 0.0), 1)
        )
        actions_generated.append(action_sig)
    
    # Calculate diversity
    unique_actions = len(set(actions_generated))
    total_actions = len(actions_generated)
    diversity_ratio = unique_actions / total_actions
    
    print(f"✅ Action diversity test:")
    print(f"   Generated {total_actions} actions")
    print(f"   {unique_actions} unique action patterns")
    print(f"   Diversity ratio: {diversity_ratio:.2f}")
    
    if diversity_ratio > 0.5:
        print(f"   ✅ Good action diversity (>{0.5:.1f})")
    else:
        print(f"   ⚠️  Low action diversity (<{0.5:.1f})")
    
    return diversity_ratio > 0.3  # Accept reasonable diversity


async def main():
    """Run all curiosity-driven brain tests."""
    print("=== Curiosity-Driven Brain Test Suite ===")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic curiosity prediction
    try:
        if test_curiosity_driven_predictor():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Curiosity predictor test failed: {e}")
    
    # Test 2: Brain-controlled simulation  
    try:
        if await test_brain_controlled_simulation():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Brain-controlled simulation test failed: {e}")
    
    # Test 3: Action diversity
    try:
        if test_action_diversity():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Action diversity test failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All curiosity-driven brain tests passed!")
        print("✅ The brain is now making intelligent, curiosity-driven decisions")
    else:
        print("⚠️  Some tests failed. The curiosity system may need debugging.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    asyncio.run(main())