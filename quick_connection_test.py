#!/usr/bin/env python3
"""
Quick Connection Test

Test connection to your running server and run a brief learning session.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent
sys.path.insert(0, str(brain_root))

from server.src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

def main():
    """Quick connection and learning test."""
    print("🔌 Quick Connection Test")
    print("=" * 30)
    
    client = MinimalBrainClient()
    environment = SensoryMotorWorld(random_seed=42)
    
    # Test connection
    print("🔗 Testing connection...")
    if not client.connect():
        print("❌ Failed to connect to brain server")
        return False
    
    print("✅ Connected successfully!")
    
    # Run 60 seconds of learning
    print("🧠 Running 60-second learning session...")
    
    experiences = []
    start_time = time.time()
    end_time = start_time + 60  # 60 seconds
    cycle_count = 0
    
    try:
        while time.time() < end_time:
            # Get sensory input
            sensory_input = environment.get_sensory_input()
            
            # Get brain action
            action_start = time.time()
            action = client.get_action(sensory_input, timeout=5.0)
            response_time = time.time() - action_start
            
            if action is None:
                print("⚠️  Timeout occurred")
                continue
            
            # Execute action
            result = environment.execute_action(action)
            
            # Record experience
            experience = {
                'cycle': cycle_count,
                'response_time_ms': response_time * 1000,
                'action': action,
                'light_distance': result.get('metrics', {}).get('min_light_distance', 1.0)
            }
            experiences.append(experience)
            
            cycle_count += 1
            
            # Brief progress report
            if cycle_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   {elapsed:.0f}s: {cycle_count} cycles, {response_time*1000:.1f}ms response")
            
            time.sleep(0.1)
        
        client.disconnect()
        
        # Quick analysis
        print("\n📊 Results:")
        print(f"   Duration: 60s")
        print(f"   Total cycles: {len(experiences)}")
        print(f"   Rate: {len(experiences)/60*60:.1f} cycles/min")
        
        if experiences:
            avg_response = sum(exp['response_time_ms'] for exp in experiences) / len(experiences)
            print(f"   Avg response: {avg_response:.1f}ms")
            
            # Check for basic learning signs
            early_responses = experiences[:len(experiences)//4]
            late_responses = experiences[-len(experiences)//4:]
            
            if early_responses and late_responses:
                early_avg = sum(exp['response_time_ms'] for exp in early_responses) / len(early_responses)
                late_avg = sum(exp['response_time_ms'] for exp in late_responses) / len(late_responses)
                
                print(f"   Performance: {early_avg:.1f}ms → {late_avg:.1f}ms")
                
                if late_avg < early_avg * 1.5:  # Less than 50% increase
                    print("   ✅ Performance stable")
                else:
                    print("   ⚠️  Performance degraded")
        
        # Check for persistence activity
        robot_memory_path = brain_root / "robot_memory"
        if robot_memory_path.exists():
            checkpoints = list(robot_memory_path.glob("**/checkpoint_*"))
            if checkpoints:
                print(f"   ✅ Persistence: {len(checkpoints)} checkpoints created")
            else:
                print("   ⚠️  Persistence: No checkpoints (need 1000 experiences)")
        
        print("\n✅ Connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        client.disconnect()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)