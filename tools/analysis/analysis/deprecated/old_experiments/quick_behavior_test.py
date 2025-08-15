#!/usr/bin/env python3
"""
Quick Behavioral Analysis Test

Addresses the fundamental question: Is the robot's cautious behavior 
authentic biological learning, or should we adjust parameters?

This will run a focused 10-minute test to analyze:
1. Whether movement patterns change over time
2. If the robot is learning or just being overly cautious
3. What the "correct" behavior should actually be
"""

import sys
import os
import time
import numpy as np
from collections import deque

# Add brain directory to path
brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, brain_dir)

from demos.test_demo import PiCarXTextDemo


def analyze_robot_behavior(duration_minutes=10):
    """
    Run focused behavioral analysis to determine if robot behavior is authentic.
    """
    print(f"🔬 Quick Behavioral Analysis ({duration_minutes} minutes)")
    print("=" * 60)
    print("Key Questions:")
    print("1. Does the robot's movement become more confident over time?")
    print("2. Is cautious behavior evidence of proper learning?")
    print("3. Should we adjust parameters or trust the process?")
    print()
    
    # Initialize demo
    demo = PiCarXTextDemo()
    
    # Tracking variables
    position_history = []
    confidence_history = []
    movement_speeds = []
    rotation_amounts = []
    
    start_time = time.time()
    step = 0
    
    print("🤖 Starting behavioral observation...")
    
    while (time.time() - start_time) < (duration_minutes * 60):
        try:
            # Execute one robot cycle
            cycle_result = demo.robot.control_cycle()
            
            # Get robot status
            status = demo.robot.get_robot_status()
            position = status['position']
            
            # Track position
            position_history.append(position)
            
            # Calculate movement metrics
            if len(position_history) >= 2:
                prev_pos = position_history[-2]
                curr_pos = position_history[-1]
                
                # Distance moved
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                movement_speeds.append(distance)
                
                # Rotation amount
                angle_diff = abs(curr_pos[2] - prev_pos[2])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                rotation_amounts.append(angle_diff)
            
            # Track confidence
            brain_stats = demo.robot.brain.get_brain_stats()
            if 'prediction_engine' in brain_stats:
                # Try to get confidence from recent predictions
                confidence = getattr(demo.robot.brain, '_last_confidence', 0.5)
                confidence_history.append(confidence)
            
            step += 1
            
            # Progress reporting every 30 seconds
            if step % 30 == 0:
                elapsed_minutes = (time.time() - start_time) / 60
                
                print(f"\n📊 Progress Report ({elapsed_minutes:.1f} min)")
                print(f"   Position: ({position[0]:.1f}, {position[1]:.1f}) @ {position[2]:.0f}°")
                print(f"   Steps completed: {step}")
                
                if movement_speeds:
                    recent_speeds = movement_speeds[-30:] if len(movement_speeds) >= 30 else movement_speeds
                    avg_speed = np.mean(recent_speeds)
                    print(f"   Average movement speed: {avg_speed:.3f} units/step")
                
                if rotation_amounts:
                    recent_rotations = rotation_amounts[-30:] if len(rotation_amounts) >= 30 else rotation_amounts
                    avg_rotation = np.mean(recent_rotations)
                    print(f"   Average rotation: {avg_rotation:.1f}°/step")
                
                if confidence_history:
                    recent_confidence = confidence_history[-30:] if len(confidence_history) >= 30 else confidence_history
                    avg_confidence = np.mean(recent_confidence)
                    print(f"   Average confidence: {avg_confidence:.3f}")
            
            # Control timing (1 step per second for realism)
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
            break
        except Exception as e:
            print(f"\n⚠️  Error at step {step}: {e}")
            continue
    
    # Final analysis
    print(f"\n🎯 BEHAVIORAL ANALYSIS RESULTS")
    print("=" * 60)
    
    # Movement analysis
    if position_history:
        start_pos = position_history[0]
        end_pos = position_history[-1]
        total_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + 
                               (end_pos[1] - start_pos[1])**2)
        
        print(f"📍 Spatial Behavior:")
        print(f"   Start position: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
        print(f"   End position: ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
        print(f"   Net displacement: {total_distance:.2f} units")
        print(f"   Total steps: {len(position_history)}")
        print(f"   Steps per unit distance: {len(position_history) / max(0.1, total_distance):.1f}")
    
    # Movement speed analysis
    if movement_speeds:
        early_speeds = movement_speeds[:min(50, len(movement_speeds)//3)]
        late_speeds = movement_speeds[-min(50, len(movement_speeds)//3):]
        
        early_avg = np.mean(early_speeds) if early_speeds else 0
        late_avg = np.mean(late_speeds) if late_speeds else 0
        
        print(f"\n🏃 Movement Evolution:")
        print(f"   Early average speed: {early_avg:.4f} units/step")
        print(f"   Late average speed: {late_avg:.4f} units/step")
        print(f"   Speed change: {((late_avg - early_avg) / max(0.001, early_avg) * 100):+.1f}%")
        
        if late_avg > early_avg * 1.2:
            print("   ✅ Robot became more confident in movement")
        elif late_avg < early_avg * 0.8:
            print("   ⚠️  Robot became more cautious over time")
        else:
            print("   ➡️  Movement speed remained consistent")
    
    # Rotation analysis
    if rotation_amounts:
        early_rotations = rotation_amounts[:min(50, len(rotation_amounts)//3)]
        late_rotations = rotation_amounts[-min(50, len(rotation_amounts)//3):]
        
        early_rot = np.mean(early_rotations) if early_rotations else 0
        late_rot = np.mean(late_rotations) if late_rotations else 0
        
        print(f"\n🔄 Rotation Behavior:")
        print(f"   Early rotation: {early_rot:.1f}°/step")
        print(f"   Late rotation: {late_rot:.1f}°/step")
        
        if late_rot < early_rot * 0.7:
            print("   ✅ Less aimless rotation over time (good sign)")
        elif late_rot > early_rot * 1.3:
            print("   ⚠️  More rotation over time (possible confusion)")
        else:
            print("   ➡️  Rotation behavior remained consistent")
    
    # Confidence analysis
    if confidence_history:
        early_conf = confidence_history[:min(50, len(confidence_history)//3)]
        late_conf = confidence_history[-min(50, len(confidence_history)//3):]
        
        early_conf_avg = np.mean(early_conf) if early_conf else 0
        late_conf_avg = np.mean(late_conf) if late_conf else 0
        
        print(f"\n🎯 Confidence Evolution:")
        print(f"   Early confidence: {early_conf_avg:.3f}")
        print(f"   Late confidence: {late_conf_avg:.3f}")
        print(f"   Confidence change: {((late_conf_avg - early_conf_avg)):+.3f}")
        
        if late_conf_avg > early_conf_avg + 0.1:
            print("   ✅ Confidence increased (learning occurring)")
        elif late_conf_avg < early_conf_avg - 0.1:
            print("   ⚠️  Confidence decreased (possible overfit)")
        else:
            print("   ➡️  Confidence remained stable")
    
    # Overall assessment
    print(f"\n🧠 BEHAVIORAL ASSESSMENT:")
    
    # Calculate behavioral authenticity
    movement_progression = False
    if movement_speeds and len(movement_speeds) > 20:
        early_avg = np.mean(movement_speeds[:len(movement_speeds)//3])
        late_avg = np.mean(movement_speeds[-len(movement_speeds)//3:])
        movement_progression = late_avg > early_avg * 1.1
    
    confidence_progression = False
    if confidence_history and len(confidence_history) > 20:
        early_avg = np.mean(confidence_history[:len(confidence_history)//3])
        late_avg = np.mean(confidence_history[-len(confidence_history)//3:])
        confidence_progression = late_avg > early_avg + 0.05
    
    if movement_progression and confidence_progression:
        print("   🟢 AUTHENTIC LEARNING: Robot shows genuine learning progression")
        print("      The cautious behavior is CORRECT - it's building confidence properly")
        recommendation = "Trust the process - this is biological learning"
    elif movement_progression or confidence_progression:
        print("   🟡 PARTIAL LEARNING: Some evidence of learning progression")
        print("      The behavior is mostly authentic but could use slight tuning")
        recommendation = "Minor parameter adjustments might help"
    else:
        print("   🔴 LIMITED LEARNING: Little evidence of learning progression")
        print("      The cautious behavior may be excessive for this environment")
        recommendation = "Consider parameter adjustments to encourage exploration"
    
    print(f"\n💡 RECOMMENDATION: {recommendation}")
    
    # Specific suggestions
    print(f"\n🛠️  SPECIFIC SUGGESTIONS:")
    
    if not movement_progression:
        print("   • Consider increasing action scaling for more assertive movement")
        print("   • Reduce autopilot confidence threshold to encourage more exploration")
    
    if total_distance < 2.0:
        print("   • Robot is very conservative - might need exploration boost")
        print("   • Consider adding slight movement bias in action selection")
    
    if len(position_history) > 0:
        efficiency = total_distance / len(position_history)
        if efficiency < 0.01:
            print("   • Very low movement efficiency - robot is mostly rotating in place")
            print("   • This might be correct for early learning, but monitor for progression")
    
    print(f"\n⏰ Analysis complete after {(time.time() - start_time)/60:.1f} minutes")
    
    return {
        'movement_progression': movement_progression,
        'confidence_progression': confidence_progression,
        'total_distance': total_distance if position_history else 0,
        'steps': len(position_history),
        'recommendation': recommendation
    }


if __name__ == "__main__":
    # Quick test first
    print("Starting 5-minute behavioral authenticity test...")
    results = analyze_robot_behavior(duration_minutes=5)
    
    print(f"\n{'='*60}")
    print("🎓 FINAL VERDICT ON ROBOT BEHAVIOR:")
    print(f"{'='*60}")
    
    if results['movement_progression'] and results['confidence_progression']:
        print("✅ The robot's cautious behavior is AUTHENTIC biological learning")
        print("   The slow, careful exploration is exactly what we'd expect")
        print("   from a system building spatial intelligence from scratch.")
    elif results['total_distance'] > 1.0:
        print("🟡 The robot shows some learning but could benefit from tuning")
        print("   Consider running longer sessions to see more progression.")
    else:
        print("🔴 The robot may be too conservative for effective learning")
        print("   Parameters might need adjustment to encourage exploration.")