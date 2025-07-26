#!/usr/bin/env python3
"""
Simple Embodied Learning Test
Test the brain with more realistic sensory patterns and motor actions
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple

# Add brain server to path
sys.path.insert(0, os.path.dirname(__file__))

from src.brain_factory import BrainFactory


def simulate_robot_sensory_data(cycle: int) -> List[float]:
    """Generate realistic robot sensory data"""
    # Simulate distance sensors (4 sensors, 0.0-1.0 range)
    # Robot moving through environment with obstacles
    distance_sensors = [
        0.8 + 0.2 * np.sin(cycle * 0.1),  # Front sensor
        0.6 + 0.3 * np.cos(cycle * 0.15),  # Right sensor  
        0.9 + 0.1 * np.sin(cycle * 0.08),  # Back sensor
        0.7 + 0.2 * np.cos(cycle * 0.12)   # Left sensor
    ]
    
    # Simulate camera/visual data (simplified to 4 values)
    visual_data = [
        0.3 + 0.4 * np.sin(cycle * 0.05),  # Brightness
        0.5 + 0.3 * np.cos(cycle * 0.07),  # Color dominance
        0.4 + 0.2 * np.sin(cycle * 0.06),  # Motion detection
        0.6 + 0.1 * np.cos(cycle * 0.04)   # Object detection
    ]
    
    # Simulate proprioceptive data (robot's internal state)
    proprioceptive = [
        0.5 + 0.1 * np.sin(cycle * 0.2),   # Battery level
        0.7 + 0.1 * np.cos(cycle * 0.18),  # Temperature
        0.4 + 0.2 * np.sin(cycle * 0.22),  # Motor load
        0.6 + 0.15 * np.cos(cycle * 0.25)  # System health
    ]
    
    # Simulate IMU data (orientation, acceleration)
    imu_data = [
        0.0 + 0.3 * np.sin(cycle * 0.03),  # Roll
        0.0 + 0.2 * np.cos(cycle * 0.035), # Pitch  
        0.5 + 0.4 * np.sin(cycle * 0.02),  # Yaw
        0.1 + 0.05 * np.cos(cycle * 0.045) # Acceleration
    ]
    
    # Combine all sensory data (16D total)
    return distance_sensors + visual_data + proprioceptive + imu_data


def evaluate_motor_action(action: List[float]) -> Dict[str, float]:
    """Evaluate how good a motor action is for robot survival/goals"""
    left_motor = action[0] if len(action) > 0 else 0.0
    right_motor = action[1] if len(action) > 1 else 0.0
    servo_angle = action[2] if len(action) > 2 else 0.0
    aux_control = action[3] if len(action) > 3 else 0.0
    
    # Evaluate action quality (simple heuristics)
    movement_balance = 1.0 - abs(left_motor - right_motor)  # Prefer balanced movement
    motor_efficiency = 1.0 - (abs(left_motor) + abs(right_motor)) / 2.0  # Prefer energy efficiency
    servo_stability = 1.0 - abs(servo_angle)  # Prefer stable servo position
    
    action_quality = (movement_balance + motor_efficiency + servo_stability) / 3.0
    
    return {
        'action_quality': action_quality,
        'movement_balance': movement_balance,
        'motor_efficiency': motor_efficiency,
        'servo_stability': servo_stability
    }


def run_embodied_learning_test(cycles: int = 200) -> Dict[str, any]:
    """Run embodied learning test with realistic sensory-motor patterns"""
    
    print("🤖 Embodied Learning Test")
    print("Simulating robot with realistic sensory-motor patterns")
    print("=" * 60)
    
    # Clear memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
        print("🗑️ Cleared robot memory for fresh embodied learning")
    
    results = {
        'cycles_completed': 0,
        'learning_progression': [],
        'action_quality_progression': [],
        'sensory_patterns': [],
        'motor_patterns': [],
        'prediction_confidence_progression': [],
        'brain_performance': {}
    }
    
    # Create brain
    brain = BrainFactory(quiet_mode=True)
    print(f"🧠 Brain initialized: {brain.brain_type} architecture")
    
    action_qualities = []
    prediction_confidences = []
    
    print(f"\n🔄 Running {cycles} embodied learning cycles...")
    
    for cycle in range(cycles):
        # Generate realistic sensory data
        sensory_data = simulate_robot_sensory_data(cycle)
        
        # Process through brain
        motor_action, brain_state = brain.process_sensory_input(sensory_data)
        
        # Evaluate motor action quality
        action_eval = evaluate_motor_action(motor_action)
        action_quality = action_eval['action_quality']
        
        # Track learning progression
        prediction_confidence = brain_state.get('prediction_confidence', 0.0)
        
        action_qualities.append(action_quality)
        prediction_confidences.append(prediction_confidence)
        
        # Store cycle data
        if cycle % 20 == 0:
            results['sensory_patterns'].append(sensory_data[:4])  # Just distance sensors
            results['motor_patterns'].append(motor_action)
            results['prediction_confidence_progression'].append(prediction_confidence)
            results['action_quality_progression'].append(action_quality)
            
            # Progress reporting
            if cycle % 40 == 0:
                avg_action_quality = np.mean(action_qualities[-20:]) if len(action_qualities) >= 20 else np.mean(action_qualities)
                avg_prediction_confidence = np.mean(prediction_confidences[-20:]) if len(prediction_confidences) >= 20 else np.mean(prediction_confidences)
                
                print(f"   Cycle {cycle:3d}: Action Quality={avg_action_quality:.3f}, Prediction Confidence={avg_prediction_confidence:.3f}")
    
    results['cycles_completed'] = cycles
    
    # Analyze learning progression
    if len(action_qualities) >= 40:
        early_action_quality = np.mean(action_qualities[:20])
        late_action_quality = np.mean(action_qualities[-20:])
        action_improvement = late_action_quality - early_action_quality
        
        early_prediction = np.mean(prediction_confidences[:20])
        late_prediction = np.mean(prediction_confidences[-20:])
        prediction_improvement = late_prediction - early_prediction
        
        results['learning_progression'] = {
            'early_action_quality': early_action_quality,
            'late_action_quality': late_action_quality,
            'action_improvement': action_improvement,
            'early_prediction_confidence': early_prediction,
            'late_prediction_confidence': late_prediction,
            'prediction_improvement': prediction_improvement
        }
        
        print(f"\n📊 Learning Analysis:")
        print(f"   Action Quality: {early_action_quality:.3f} → {late_action_quality:.3f} (Δ{action_improvement:+.3f})")
        print(f"   Prediction Confidence: {early_prediction:.3f} → {late_prediction:.3f} (Δ{prediction_improvement:+.3f})")
        
        # Overall embodied learning score
        normalized_action_improvement = max(0, action_improvement * 2.0)  # Scale to 0-1
        normalized_prediction_improvement = max(0, prediction_improvement)
        embodied_learning_score = (normalized_action_improvement + normalized_prediction_improvement) / 2.0
        
        results['embodied_learning_score'] = embodied_learning_score
        print(f"   Embodied Learning Score: {embodied_learning_score:.3f}")
        
        if embodied_learning_score > 0.1:
            print("   ✅ EMBODIED LEARNING DETECTED!")
        else:
            print("   ⚠️  Limited embodied learning observed")
    
    # Get final brain statistics
    brain_stats = brain.get_brain_stats()
    results['brain_performance'] = brain_stats
    
    brain.finalize_session()
    
    return results


if __name__ == "__main__":
    results = run_embodied_learning_test(200)
    print(f"\n🎯 Embodied Learning Test Complete!")
    
    if 'embodied_learning_score' in results:
        score = results['embodied_learning_score']
        if score > 0.2:
            print(f"🎉 EXCELLENT embodied learning: {score:.3f}")
        elif score > 0.1:
            print(f"✅ GOOD embodied learning: {score:.3f}")
        elif score > 0.05:
            print(f"⚠️  MODERATE embodied learning: {score:.3f}")
        else:
            print(f"❌ LIMITED embodied learning: {score:.3f}")
    else:
        print("❌ Unable to analyze embodied learning")