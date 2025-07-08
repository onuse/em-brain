#!/usr/bin/env python3
"""
Demonstration of curiosity-driven brain in 2D world simulation.
Shows the robot making intelligent, curiosity-driven decisions in real-time.
"""

import asyncio
import time
from simulation.brainstem_sim import GridWorldBrainstem
from core.brain_interface import BrainInterface
from predictor.triple_predictor import TriplePredictor
from core.world_graph import WorldGraph


def print_robot_status(brainstem, step, prediction=None):
    """Print current robot status."""
    stats = brainstem.get_simulation_stats()
    robot_pos = stats['robot_position']
    
    print(f"Step {step:2d}: "
          f"Pos=({robot_pos[0]:2d},{robot_pos[1]:2d}) "
          f"Health={stats['robot_health']:.2f} "
          f"Energy={stats['robot_energy']:.2f} "
          f"Orientation={stats['robot_orientation']}")
    
    if prediction:
        action = prediction.motor_action
        print(f"        Action: forward={action.get('forward_motor', 0.0):.2f} "
              f"turn={action.get('turn_motor', 0.0):.2f} "
              f"brake={action.get('brake_motor', 0.0):.2f}")
        print(f"        Decision: {prediction.consensus_strength} "
              f"(confidence={prediction.confidence:.2f})")


async def run_curiosity_driven_simulation():
    """Run a curiosity-driven simulation with detailed output."""
    print("🧠 Curiosity-Driven Robot in 2D World")
    print("=====================================")
    
    # Create larger world for more interesting behavior
    brainstem = GridWorldBrainstem(
        world_width=12,
        world_height=12,
        seed=42,  # Deterministic for consistent demo
        use_sockets=False
    )
    
    # Create brain with high exploration rate for demo
    predictor = TriplePredictor(
        base_time_budget=0.05,   # Allow more thinking time
        exploration_rate=0.7     # High curiosity/exploration
    )
    brain = BrainInterface(predictor)
    brainstem.brain_client = brain  # Use our curiosity-driven brain
    
    # Display initial world state
    world_state = brainstem.get_world_state()
    print(f"World size: {world_state['world_size']}")
    print(f"Robot starting position: {world_state['robot_position']}")
    print(f"Robot starting orientation: {world_state['robot_orientation']}")
    print()
    
    # Run simulation step by step
    print("🏃 Running Curiosity-Driven Simulation...")
    print("(Robot will explore based on prediction uncertainty)")
    print()
    
    step_count = 0
    max_steps = 25
    
    try:
        while step_count < max_steps and brainstem.is_robot_alive():
            # Get current sensor readings
            sensor_packet = brainstem.get_sensor_readings()
            
            # Calculate mental context (robot's self-awareness)
            robot_stats = brainstem.get_simulation_stats()
            robot_pos = robot_stats['robot_position']
            mental_context = [
                float(robot_pos[0]) / brainstem.simulation.width,    # normalized x
                float(robot_pos[1]) / brainstem.simulation.height,  # normalized y
                robot_stats['robot_energy'],                        # energy level
                robot_stats['robot_health'],                        # health level
                float(step_count) / max_steps                       # mission progress
            ]
            
            # Determine threat level based on robot state
            if robot_stats['robot_health'] < 0.3 or robot_stats['robot_energy'] < 0.2:
                threat_level = "danger"
            elif robot_stats['robot_health'] < 0.6 or robot_stats['robot_energy'] < 0.5:
                threat_level = "alert"  
            else:
                threat_level = "normal"
            
            # Get brain's curiosity-driven decision
            prediction = brain.process_sensory_input(sensor_packet, mental_context, threat_level)
            
            # Print status before action
            print_robot_status(brainstem, step_count, prediction)
            
            # Execute brain's decision
            is_alive = brainstem.execute_prediction_packet(prediction)
            
            if not is_alive:
                print("💀 Robot died!")
                break
            
            step_count += 1
            
            # Brief pause to make demo readable
            await asyncio.sleep(0.2)
            
            # Show learning progress every 5 steps
            if step_count % 5 == 0:
                brain_stats = brain.get_brain_statistics()
                experiences = brain_stats['interface_stats']['total_experiences']
                exploration_rate = predictor.exploration_rate
                print(f"     📚 Brain has {experiences} experiences, "
                      f"exploration rate: {exploration_rate:.2f}")
                print()
    
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    
    # Final summary
    print("\n📊 Final Results:")
    print("================")
    
    final_stats = brainstem.get_simulation_stats()
    brain_stats = brain.get_brain_statistics()
    
    print(f"Steps completed: {step_count}")
    print(f"Final robot health: {final_stats['robot_health']:.2f}")
    print(f"Final robot energy: {final_stats['robot_energy']:.2f}")
    print(f"Final position: {final_stats['robot_position']}")
    print(f"Total collisions: {final_stats['total_collisions']}")
    print(f"Food consumed: {final_stats['total_food_consumed']}")
    
    print(f"\nBrain learning:")
    print(f"Experiences accumulated: {brain_stats['interface_stats']['total_experiences']}")
    print(f"Total predictions made: {brain_stats['predictor_stats']['total_predictions']}")
    print(f"Final exploration rate: {predictor.exploration_rate:.2f}")
    
    if 'predictor_stats' in brain_stats:
        predictor_stats = brain_stats['predictor_stats']
        if 'consensus_breakdown' in predictor_stats:
            consensus_stats = predictor_stats['consensus_breakdown']
            print(f"Decision patterns: {consensus_stats}")
        else:
            print(f"Prediction stats: {predictor_stats}")
    
    return step_count, final_stats, brain_stats


async def compare_random_vs_curiosity():
    """Compare random exploration vs curiosity-driven exploration."""
    print("\n🆚 Comparison: Random vs Curiosity-Driven")
    print("=========================================")
    
    results = {}
    
    for mode in ["random", "curiosity"]:
        print(f"\n🔄 Running {mode.upper()} mode...")
        
        brainstem = GridWorldBrainstem(
            world_width=10, world_height=10, seed=42, use_sockets=False
        )
        
        if mode == "curiosity":
            # Curiosity-driven brain
            predictor = TriplePredictor(
                base_time_budget=0.02,
                exploration_rate=0.6
            )
            brain = BrainInterface(predictor)
            brainstem.brain_client = brain
            
            # Run brain-controlled simulation
            sim_results = await brainstem.run_brain_controlled_simulation(
                steps=20, step_delay=0.05
            )
            
        else:
            # Random exploration baseline
            sim_results = {
                'steps_completed': 0,
                'final_robot_state': brainstem.get_simulation_stats()
            }
            
            # Simulate random exploration
            for _ in range(20):
                import random
                random_action = {
                    'forward_motor': random.uniform(-0.3, 0.7),
                    'turn_motor': random.uniform(-0.5, 0.5),
                    'brake_motor': random.uniform(0.0, 0.2)
                }
                
                is_alive = brainstem.execute_motor_commands(random_action)
                if not is_alive:
                    break
                sim_results['steps_completed'] += 1
            
            sim_results['final_robot_state'] = brainstem.get_simulation_stats()
        
        results[mode] = sim_results
        
        print(f"   Steps: {sim_results['steps_completed']}")
        print(f"   Final health: {sim_results['final_robot_state']['robot_health']:.2f}")
        print(f"   Final energy: {sim_results['final_robot_state']['robot_energy']:.2f}")
        print(f"   Collisions: {sim_results['final_robot_state']['total_collisions']}")
        print(f"   Food found: {sim_results['final_robot_state']['total_food_consumed']}")
    
    # Compare results
    print(f"\n🏆 Comparison Results:")
    random_steps = results['random']['steps_completed']
    curiosity_steps = results['curiosity']['steps_completed']
    
    if curiosity_steps > random_steps:
        improvement = ((curiosity_steps - random_steps) / random_steps * 100)
        print(f"   ✅ Curiosity-driven robot lasted {improvement:.1f}% longer")
    else:
        print(f"   ⚠️  Random exploration performed better this time")
    
    random_health = results['random']['final_robot_state']['robot_health']
    curiosity_health = results['curiosity']['final_robot_state']['robot_health']
    
    if curiosity_health > random_health:
        print(f"   ✅ Curiosity-driven robot maintained better health")
    
    print(f"   🧠 Curiosity-driven robot made intelligent decisions based on prediction uncertainty")


async def main():
    """Main demonstration function."""
    print("🤖 Emergent Intelligence: Curiosity-Driven Robot Demo")
    print("=====================================================")
    print("This demonstrates a robot making intelligent decisions")
    print("based on curiosity (prediction uncertainty) rather than")
    print("random exploration or hardcoded behaviors.")
    print()
    
    # Run main curiosity-driven simulation
    await run_curiosity_driven_simulation()
    
    # Run comparison
    await compare_random_vs_curiosity()
    
    print("\n✨ Summary:")
    print("The robot uses prediction errors to drive exploration:")
    print("• High prediction error → more exploration (curious)")  
    print("• Low prediction error → more exploitation (confident)")
    print("• Actions are chosen to maximize learning potential")
    print("• Intelligence emerges from embodied experience")
    print("\n🎉 Curiosity-driven emergent intelligence demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())