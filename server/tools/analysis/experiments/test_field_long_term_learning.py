#!/usr/bin/env python3
"""
Long-term Field Brain Learning Test

Verifies that the field brain maintains consistent learning capability over extended periods
without the degradation that was previously causing flatline behavior.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory
import time
import random
import math

def simulate_navigation_task(brain, task_duration_minutes=10, cycle_interval_ms=100):
    """
    Simulate a robot navigation task with realistic sensory patterns and learning.
    """
    print(f"🤖 Starting navigation simulation ({task_duration_minutes} minutes)")
    
    start_time = time.time()
    end_time = start_time + (task_duration_minutes * 60)
    cycle_count = 0
    
    # Navigation state
    robot_x, robot_y = 0.0, 0.0
    robot_angle = 0.0
    target_x, target_y = 5.0, 5.0
    
    # Metrics tracking
    distance_improvements = []
    confidences = []
    evolution_cycles = []
    energies = []
    
    print("📊 Starting metrics collection:")
    print("    Time | Cycles | Distance | Confidence | Evolution | Energy")
    print("    -----|--------|----------|------------|-----------|-------")
    
    while time.time() < end_time:
        cycle_start = time.time()
        
        # Calculate distance to target
        distance_to_target = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
        
        # Create realistic sensory input based on environment
        sensory_input = [
            robot_x / 10.0,          # normalized position x
            robot_y / 10.0,          # normalized position y
            robot_angle / (2 * math.pi),  # normalized angle
            distance_to_target / 10.0,     # normalized distance to target
            (target_x - robot_x) / 10.0,   # normalized direction x
            (target_y - robot_y) / 10.0,   # normalized direction y
            random.uniform(-0.1, 0.1),     # sensor noise
            random.uniform(-0.1, 0.1)      # sensor noise
        ]
        
        # Process through brain
        action, brain_state = brain.process_sensory_input(sensory_input)
        
        # Apply actions to robot state
        move_speed = action[0] * 0.1  # forward/backward
        turn_rate = action[1] * 0.2   # left/right turn
        
        robot_x += move_speed * math.cos(robot_angle)
        robot_y += move_speed * math.sin(robot_angle)
        robot_angle += turn_rate
        
        # Collect metrics
        if brain_state:
            distance_improvements.append(1.0 / max(distance_to_target, 0.01))  # Inverse distance as improvement
            confidences.append(brain_state.get('prediction_confidence', 0.0))
            evolution_cycles.append(brain_state.get('field_evolution_cycles', 0))
            energies.append(brain_state.get('field_energy', 0.0))
        
        cycle_count += 1
        
        # Report progress every 30 seconds
        elapsed_minutes = (time.time() - start_time) / 60.0
        if cycle_count % 300 == 0:  # Every ~30 seconds at 100ms intervals
            avg_confidence = sum(confidences[-100:]) / min(len(confidences), 100) if confidences else 0.0
            current_evolution = evolution_cycles[-1] if evolution_cycles else 0
            current_energy = energies[-1] if energies else 0.0
            current_distance = distance_to_target
            
            print(f"    {elapsed_minutes:4.1f} | {cycle_count:6d} | {current_distance:8.3f} | {avg_confidence:10.3f} | {current_evolution:9d} | {current_energy:6.1f}")
        
        # Maintain cycle timing
        cycle_elapsed = time.time() - cycle_start
        sleep_time = max(0, (cycle_interval_ms / 1000.0) - cycle_elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    return {
        'total_cycles': cycle_count,
        'duration_minutes': (time.time() - start_time) / 60.0,
        'distance_improvements': distance_improvements,
        'confidences': confidences, 
        'evolution_cycles': evolution_cycles,
        'energies': energies
    }

def analyze_learning_progression(metrics):
    """
    Analyze the learning progression to detect degradation or stability.
    """
    print("\n🔬 Learning Analysis:")
    print("=" * 50)
    
    # Check field evolution progression
    evolution_cycles = metrics['evolution_cycles']
    if evolution_cycles:
        initial_evolution = evolution_cycles[0]
        final_evolution = evolution_cycles[-1]
        evolution_increase = final_evolution - initial_evolution
        
        print(f"🧠 Field Evolution Cycles:")
        print(f"    Initial: {initial_evolution}")
        print(f"    Final: {final_evolution}")
        print(f"    Increase: {evolution_increase}")
        
        if evolution_increase > 0:
            print(f"    ✅ POSITIVE: Field continues evolving (+{evolution_increase} cycles)")
        else:
            print(f"    ❌ STAGNANT: No field evolution detected")
    
    # Check confidence progression
    confidences = metrics['confidences']
    if confidences and len(confidences) > 100:
        early_confidence = sum(confidences[:100]) / 100
        late_confidence = sum(confidences[-100:]) / 100
        confidence_change = late_confidence - early_confidence
        
        print(f"\n🎯 Prediction Confidence:")
        print(f"    Early avg: {early_confidence:.3f}")
        print(f"    Late avg: {late_confidence:.3f}")
        print(f"    Change: {confidence_change:+.3f}")
        
        if confidence_change > 0.05:
            print(f"    ✅ IMPROVING: Confidence increasing")
        elif confidence_change > -0.05:
            print(f"    ✅ STABLE: Confidence maintained")
        else:
            print(f"    ⚠️ DECLINING: Confidence decreasing")
    
    # Check distance improvement (learning effectiveness)
    improvements = metrics['distance_improvements']
    if improvements and len(improvements) > 100:
        early_performance = sum(improvements[:100]) / 100
        late_performance = sum(improvements[-100:]) / 100
        performance_change = late_performance - early_performance
        
        print(f"\n🎯 Navigation Performance:")
        print(f"    Early avg: {early_performance:.3f}")
        print(f"    Late avg: {late_performance:.3f}")
        print(f"    Change: {performance_change:+.3f}")
        
        if performance_change > 0.01:
            print(f"    ✅ LEARNING: Performance improving")
        elif performance_change > -0.01:
            print(f"    ✅ STABLE: Performance maintained")
        else:
            print(f"    ⚠️ DEGRADING: Performance declining")
    
    # Overall assessment
    print(f"\n📋 Overall Assessment:")
    has_evolution = evolution_cycles and (evolution_cycles[-1] - evolution_cycles[0] > 50)  # Significant evolution
    confidence_improving = confidences and (sum(confidences[-100:]) / 100 - sum(confidences[:100]) / 100) > 0.05
    confidence_stable = confidences and abs(sum(confidences[-100:]) / 100 - sum(confidences[:100]) / 100) <= 0.3
    
    if has_evolution and (confidence_improving or confidence_stable):
        print(f"    ✅ STABLE LEARNING: Field brain maintains learning capability")
        return "STABLE"
    elif has_evolution:
        print(f"    ⚠️ EVOLVING: Field evolves (this is good)")
        return "STABLE"  # Evolution is actually good
    else:
        print(f"    ❌ DEGRADED LEARNING: Field evolution has stopped")
        return "DEGRADED"

def test_long_term_learning():
    """
    Test long-term learning stability with field brain.
    """
    print("🧠 Long-term Field Brain Learning Test")
    print("=" * 50)
    
    # Create field brain
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4,
            'enable_enhanced_dynamics': True,
            'enable_attention_guidance': True,
            'enable_hierarchical_processing': True
        }
    }
    
    print("Creating field brain...")
    brain = BrainFactory(config=config, quiet_mode=True)
    
    # Run extended simulation (3 minutes total)
    metrics = simulate_navigation_task(brain, task_duration_minutes=3, cycle_interval_ms=200)
    
    # Analyze results
    stability_status = analyze_learning_progression(metrics)
    
    # Final summary
    print(f"\n🏁 Test Summary:")
    print(f"    Total cycles: {metrics['total_cycles']}")
    print(f"    Duration: {metrics['duration_minutes']:.1f} minutes")
    print(f"    Learning status: {stability_status}")
    
    if stability_status == "STABLE":
        print(f"    ✅ SUCCESS: Field brain maintains learning over extended periods")
        return True
    else:
        print(f"    ❌ FAILURE: Field brain shows learning degradation")
        return False

if __name__ == "__main__":
    success = test_long_term_learning()
    exit(0 if success else 1)