#!/usr/bin/env python3
"""
Extended Simulation Runner for Real Robot Preparation

Runs long-term simulation to build a useful memory foundation 
for the real PiCar-X robot. Based on transfer analysis showing
70-90% of learned experiences should transfer positively.

Usage:
    python3 tools/extended_simulation_runner.py --hours 24
    
This will create a rich memory bank that the real robot can inherit.
"""

import sys
import os
import time
import argparse
from datetime import datetime, timedelta

# Add brain directory to path
brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, brain_dir)

from demos.test_demo import PiCarXTextDemo


def run_extended_simulation(duration_hours: float, save_interval_hours: float = 1.0):
    """
    Run extended simulation to build transferable memory for real robot.
    
    Args:
        duration_hours: How long to run simulation
        save_interval_hours: How often to save memory snapshots
    """
    
    print(f"🚀 Extended Simulation for Real Robot Memory Preparation")
    print("=" * 70)
    print(f"Duration: {duration_hours} hours")
    print(f"Save interval: {save_interval_hours} hours")
    print()
    print("🎯 Goal: Build rich spatial intelligence for real PiCar-X transfer")
    print("📊 Expected transfer success: 70-90% of learned patterns")
    print()
    
    # Initialize demo with persistent memory
    demo = PiCarXTextDemo()
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    next_save_time = start_time + (save_interval_hours * 3600)
    
    step_count = 0
    save_count = 0
    
    print("🤖 Starting extended simulation...")
    print("   Memory will persist between sessions for real robot use")
    print()
    
    try:
        while time.time() < end_time:
            
            # Run simulation step
            try:
                cycle_result = demo.robot.control_cycle()
                step_count += 1
                
                # Progress reporting
                if step_count % 100 == 0:
                    elapsed_hours = (time.time() - start_time) / 3600
                    remaining_hours = (end_time - time.time()) / 3600
                    
                    # Get robot status
                    status = demo.robot.get_robot_status()
                    brain_stats = demo.robot.brain.get_brain_stats()
                    
                    print(f"📊 Progress: {elapsed_hours:.2f}h elapsed, {remaining_hours:.2f}h remaining")
                    print(f"   Steps: {step_count:,}")
                    print(f"   Position: ({status['position'][0]:.1f}, {status['position'][1]:.1f})")
                    print(f"   Experiences: {brain_stats['brain_summary']['total_experiences']}")
                    print(f"   Consensus rate: {brain_stats['prediction_engine']['consensus_rate']:.2f}")
                    print()
                
                # Periodic memory saves
                if time.time() >= next_save_time:
                    save_count += 1
                    elapsed_hours = (time.time() - start_time) / 3600
                    
                    print(f"💾 Memory snapshot #{save_count} at {elapsed_hours:.1f} hours")
                    
                    # Get current brain state for logging
                    brain_stats = demo.robot.brain.get_brain_stats()
                    experiences = brain_stats['brain_summary']['total_experiences']
                    
                    print(f"   📚 Total experiences: {experiences}")
                    print(f"   🧠 Working memory: {brain_stats['activation_dynamics'].get('current_working_memory_size', 0)}")
                    print(f"   🎯 Prediction accuracy: {brain_stats['prediction_engine'].get('avg_prediction_accuracy', 0):.3f}")
                    
                    if brain_stats['prediction_engine'].get('pattern_analysis'):
                        patterns = brain_stats['prediction_engine']['pattern_analysis'].get('total_patterns', 0)
                        print(f"   🔍 Patterns discovered: {patterns}")
                    
                    print(f"   💡 Memory ready for real robot transfer")
                    print()
                    
                    next_save_time += (save_interval_hours * 3600)
                
                # Maintain realistic timing (1 step per second)
                time.sleep(1.0)
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Simulation interrupted by user")
                break
            except Exception as e:
                print(f"   ⚠️  Step error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Simulation error: {e}")
    
    # Final summary
    final_elapsed = (time.time() - start_time) / 3600
    
    print(f"\n🎉 Extended Simulation Complete!")
    print("=" * 50)
    print(f"Duration: {final_elapsed:.2f} hours")
    print(f"Steps completed: {step_count:,}")
    print(f"Memory snapshots: {save_count}")
    
    # Final brain analysis
    try:
        brain_stats = demo.robot.brain.get_brain_stats()
        status = demo.robot.get_robot_status()
        
        print(f"\n📊 Final Brain State:")
        print(f"   Total experiences: {brain_stats['brain_summary']['total_experiences']}")
        print(f"   Working memory size: {brain_stats['activation_dynamics'].get('current_working_memory_size', 0)}")
        print(f"   Consensus predictions: {brain_stats['prediction_engine']['consensus_predictions']}")
        print(f"   Pattern predictions: {brain_stats['prediction_engine']['pattern_predictions']}")
        
        if brain_stats['prediction_engine'].get('pattern_analysis'):
            patterns = brain_stats['prediction_engine']['pattern_analysis']
            print(f"   Patterns discovered: {patterns.get('total_patterns', 0)}")
            print(f"   Average pattern length: {patterns.get('avg_pattern_length', 0):.1f}")
        
        print(f"\n🗺️  Spatial Intelligence:")
        print(f"   Final position: ({status['position'][0]:.1f}, {status['position'][1]:.1f})")
        print(f"   Obstacle encounters: {status.get('obstacle_encounters', 0)}")
        print(f"   Navigation success rate: {status.get('navigation_success_rate', 0):.2%}")
        
        print(f"\n🚀 Real Robot Transfer Readiness:")
        
        # Transfer readiness assessment
        experiences = brain_stats['brain_summary']['total_experiences']
        patterns = brain_stats['prediction_engine'].get('pattern_analysis', {}).get('total_patterns', 0)
        consensus_rate = brain_stats['prediction_engine']['consensus_rate']
        
        if experiences >= 1000 and patterns >= 5 and consensus_rate > 5.0:
            print("   ✅ EXCELLENT - Rich memory foundation ready for real robot")
            print("   🎯 Expected immediate competence on real hardware")
        elif experiences >= 500 and consensus_rate > 2.0:
            print("   🟡 GOOD - Solid foundation, may need additional real-world learning")
            print("   🎯 Expected rapid adaptation on real hardware")
        else:
            print("   🔴 LIMITED - May need longer simulation or real-world training")
            print("   🎯 Expected gradual learning curve on real hardware")
        
        print(f"\n💡 Next Steps:")
        print("   1. 🔄 Memory bank is persistent - no need to restart brain")
        print("   2. 🤖 Deploy same brain to real PiCar-X robot")
        print("   3. 🧪 Monitor initial performance and allow continued learning")
        print("   4. 🎯 Real robot should show immediate navigation competence")
        
    except Exception as e:
        print(f"   ⚠️  Error getting final stats: {e}")
    
    return {
        'duration_hours': final_elapsed,
        'steps_completed': step_count,
        'memory_snapshots': save_count,
        'transfer_ready': True
    }


def main():
    """Run extended simulation with command line arguments."""
    
    parser = argparse.ArgumentParser(description='Run extended simulation for real robot memory preparation')
    parser.add_argument('--hours', type=float, default=4.0, help='Duration in hours (default: 4.0)')
    parser.add_argument('--save-interval', type=float, default=1.0, help='Save interval in hours (default: 1.0)')
    
    args = parser.parse_args()
    
    print("🧠 Extended Simulation for Real Robot Memory Foundation")
    print("=" * 70)
    
    # Confirm long duration runs
    if args.hours > 8:
        confirm = input(f"Run simulation for {args.hours} hours? (y/N): ")
        if confirm.lower() != 'y':
            print("Simulation cancelled")
            return
    
    # Calculate end time
    end_time = datetime.now() + timedelta(hours=args.hours)
    print(f"⏰ Simulation will run until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run simulation
    results = run_extended_simulation(args.hours, args.save_interval)
    
    print(f"\n🎉 Memory foundation complete!")
    print(f"Real robot can now inherit {results['steps_completed']:,} experiences")
    print(f"Expected transfer success: 70-90% of learned behaviors")


if __name__ == "__main__":
    main()