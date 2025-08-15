#!/usr/bin/env python3
"""
Quick Performance Scaling Test

Rapid test to determine if response time scales linearly with experience count.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent
sys.path.insert(0, str(brain_root))

from server.src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

def main():
    """Quick test of performance scaling."""
    print("⚡ Quick Performance Scaling Test")
    print("=" * 40)
    print("Testing: Does response time scale linearly with experience count?")
    print()
    
    client = MinimalBrainClient()
    environment = SensoryMotorWorld(random_seed=42)
    
    if not client.connect():
        print("❌ Failed to connect to brain")
        return False
    
    print("✅ Connected - running quick scaling test...")
    
    # Collect data at different experience levels
    measurements = []
    target_cycles = [10, 50, 100, 200, 300]  # Test at different experience levels
    
    try:
        cycle_count = 0
        
        for target in target_cycles:
            # Run cycles until we reach target
            while cycle_count < target:
                sensory_input = environment.get_sensory_input()
                
                # Measure response time
                start_time = time.time()
                action = client.get_action(sensory_input, timeout=10.0)
                response_time = time.time() - start_time
                
                if action is None:
                    continue
                
                environment.execute_action(action)
                cycle_count += 1
                
                # Record measurement at target
                if cycle_count == target:
                    est_experiences = max(0, cycle_count - 1)  # First cycle creates no experience
                    measurements.append({
                        'cycle': cycle_count,
                        'experiences': est_experiences,
                        'response_time_ms': response_time * 1000
                    })
                    
                    print(f"📊 Cycle {cycle_count}: {est_experiences} experiences, "
                          f"{response_time*1000:.1f}ms response")
                    break
                
                time.sleep(0.05)  # Faster cycling
        
        client.disconnect()
        
        # Analyze scaling
        print("\n📈 Scaling Analysis:")
        print("-" * 30)
        
        if len(measurements) >= 3:
            experiences = np.array([m['experiences'] for m in measurements])
            response_times = np.array([m['response_time_ms'] for m in measurements])
            
            # Calculate correlation
            correlation = np.corrcoef(experiences, response_times)[0, 1]
            
            # Fit linear regression
            linear_fit = np.polyfit(experiences, response_times, 1)
            slope_ms_per_experience = linear_fit[0]
            intercept_ms = linear_fit[1]
            
            print(f"Correlation: {correlation:.3f}")
            print(f"Slope: {slope_ms_per_experience:.3f} ms per experience")
            print(f"Intercept: {intercept_ms:.1f} ms")
            
            # Assess scaling behavior
            print(f"\n🎯 Assessment:")
            if abs(correlation) > 0.7:
                if slope_ms_per_experience > 0.1:
                    print("   ⚠️  STRONG LINEAR DEGRADATION")
                    print("   Response time will continue increasing with experience count")
                    
                    # Project future performance
                    for exp_count in [1000, 5000, 10000]:
                        projected_time = slope_ms_per_experience * exp_count + intercept_ms
                        print(f"   At {exp_count:,} experiences: {projected_time:.0f}ms")
                        
                elif slope_ms_per_experience > 0.01:
                    print("   ⚠️  MODERATE LINEAR GROWTH")
                    print("   Response time grows slowly with experience count")
                else:
                    print("   ✅ MINIMAL LINEAR GROWTH")
                    print("   Response time essentially stable")
            else:
                print("   ✅ NO LINEAR CORRELATION")
                print("   Response time appears to stabilize")
            
            # Compare first and last measurements
            first_time = measurements[0]['response_time_ms']
            last_time = measurements[-1]['response_time_ms']
            first_exp = measurements[0]['experiences']
            last_exp = measurements[-1]['experiences']
            
            print(f"\nFirst measurement: {first_time:.1f}ms ({first_exp} experiences)")
            print(f"Last measurement: {last_time:.1f}ms ({last_exp} experiences)")
            print(f"Performance change: {last_time - first_time:+.1f}ms over {last_exp - first_exp} experiences")
            
            # Rate of change
            if last_exp > first_exp:
                rate = (last_time - first_time) / (last_exp - first_exp)
                print(f"Rate of change: {rate:.3f} ms per experience")
                
                if rate < 0.01:
                    print("✅ PERFORMANCE STABLE - Rate of change very low")
                elif rate < 0.1:
                    print("⚠️  PERFORMANCE SLOWLY DEGRADING - Manageable rate")
                else:
                    print("❌ PERFORMANCE RAPIDLY DEGRADING - High rate of change")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        client.disconnect()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)