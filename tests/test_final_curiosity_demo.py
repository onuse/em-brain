#!/usr/bin/env python3
"""
Final verification that curiosity-driven robot works correctly.
"""

import asyncio
from simulation.brainstem_sim import GridWorldBrainstem
from core.brain_interface import BrainInterface
from predictor.triple_predictor import TriplePredictor


async def test_curiosity_robot():
    """Test curiosity-driven robot behavior."""
    print("🧠 Final Curiosity-Driven Robot Test")
    print("===================================")
    
    # Create robot with curiosity brain
    brainstem = GridWorldBrainstem(
        world_width=8, world_height=8, seed=42, use_sockets=False
    )
    
    predictor = TriplePredictor(
        base_time_budget=0.02,
        exploration_rate=0.6
    )
    brain = BrainInterface(predictor)
    brainstem.brain_client = brain
    
    print(f"Initial position: {brainstem.get_simulation_stats()['robot_position']}")
    
    # Run brain-controlled simulation
    results = await brainstem.run_brain_controlled_simulation(
        steps=10, step_delay=0.1
    )
    
    print(f"\n✅ Results:")
    print(f"Steps completed: {results['steps_completed']}")
    print(f"Predictions received: {results['predictions_received']}")
    print(f"Communication errors: {results['communication_errors']}")
    final_state = results['final_robot_state']
    print(f"Available keys: {list(final_state.keys())}")
    print(f"Final robot health: {final_state.get('robot_health', final_state.get('health', 'unknown'))}")
    print(f"Final robot energy: {final_state.get('robot_energy', final_state.get('energy', 'unknown'))}")
    print(f"Final position: {final_state.get('robot_position', final_state.get('position', 'unknown'))}")
    
    # Analyze decision types
    if results['performance_stats']:
        decisions = {}
        for stat in results['performance_stats']:
            consensus = stat['consensus_strength']
            decisions[consensus] = decisions.get(consensus, 0) + 1
        
        print(f"Decision types: {decisions}")
        
        curiosity_decisions = decisions.get('curiosity_driven', 0)
        total_decisions = len(results['performance_stats'])
        
        if curiosity_decisions > 0:
            print(f"✅ {curiosity_decisions}/{total_decisions} decisions were curiosity-driven")
            print("🧠 The robot is making intelligent, curiosity-driven decisions!")
            return True
        else:
            print("⚠️  No curiosity-driven decisions detected")
            return False
    else:
        print("❌ No performance stats available")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_curiosity_robot())
    if success:
        print("\n🎉 Curiosity-driven emergent intelligence working correctly!")
    else:
        print("\n❌ Curiosity system needs debugging")