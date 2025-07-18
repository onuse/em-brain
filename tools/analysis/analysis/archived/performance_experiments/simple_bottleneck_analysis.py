#!/usr/bin/env python3
"""
Simple Bottleneck Analysis

Use Python's cProfile to identify where time is spent during learning.
"""

import sys
import os
import time
import cProfile
import pstats
import io

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def run_learning_simulation():
    """Run a learning simulation to profile."""
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        quiet_mode=True
    )
    
    # Learn 100 experiences
    for i in range(100):
        sensory = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01, 0.4 + i * 0.01]
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    brain.finalize_session()


def analyze_bottlenecks():
    """Analyze learning bottlenecks using profiling."""
    print("🔬 BOTTLENECK ANALYSIS WITH PROFILING")
    print("=" * 45)
    print("Running 100 experience learning simulation with profiler...")
    print()
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run simulation with profiling
    start_time = time.time()
    profiler.enable()
    
    run_learning_simulation()
    
    profiler.disable()
    total_time = time.time() - start_time
    
    print(f"Total simulation time: {total_time:.2f}s")
    print()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    profile_output = s.getvalue()
    
    # Extract key insights
    print("🎯 TOP TIME-CONSUMING FUNCTIONS:")
    print("-" * 35)
    
    lines = profile_output.split('\n')
    header_found = False
    count = 0
    
    for line in lines:
        if 'cumulative' in line and 'percall' in line:
            header_found = True
            print(line)
            continue
        
        if header_found and line.strip() and count < 15:
            # Filter for brain-related functions
            if any(keyword in line for keyword in ['brain', 'similarity', 'activation', 'experience', 'pattern', 'store']):
                print(line)
                count += 1
    
    print()
    print("🔍 BOTTLENECK SUMMARY:")
    print("-" * 22)
    
    # Look for specific patterns in the profile
    profile_text = profile_output.lower()
    
    if 'similarity' in profile_text:
        print("✓ Similarity-related functions found in profile")
    if 'activation' in profile_text:
        print("✓ Activation-related functions found in profile")
    if 'pattern' in profile_text:
        print("✓ Pattern analysis functions found in profile")
    if 'gpu' in profile_text or 'tensor' in profile_text:
        print("✓ GPU/tensor operations found in profile")
    
    print()
    print("💡 OPTIMIZATION STRATEGY:")
    print("-" * 27)
    print("Based on the profile above, focus optimization on:")
    print("1. Functions with highest cumulative time")
    print("2. Functions called most frequently") 
    print("3. Functions with high per-call time")
    print()
    print("Consider:")
    print("• Batch processing for frequently called functions")
    print("• Lazy evaluation for expensive operations")
    print("• Caching for repeated computations")
    print("• Background/async processing for non-critical updates")


if __name__ == "__main__":
    analyze_bottlenecks()