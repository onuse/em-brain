#!/usr/bin/env python3
"""
Quick behavioral test with configurable time limit

This runs a subset of tests that can complete quickly
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

def quick_prediction_test(brain: BrainFactory, max_seconds: float = 5) -> float:
    """Quick prediction learning test"""
    print("📊 Testing prediction learning...")
    
    pattern = [0.5, 0.8, 0.3, 0.6] * 4  # 16D repeating pattern
    start_time = time.time()
    predictions = []
    cycles = 0
    
    while time.time() - start_time < max_seconds:
        # Present pattern
        action, brain_state = brain.process_sensory_input(pattern)
        
        # Check prediction confidence
        pred_conf = brain_state.get('prediction_confidence', 0.0)
        predictions.append(pred_conf)
        
        cycles += 1
        if cycles % 10 == 0:
            print(f"   Cycle {cycles}: prediction confidence = {pred_conf:.3f}")
    
    # Score: improvement in prediction confidence
    if len(predictions) > 20:
        early_conf = np.mean(predictions[:10])
        late_conf = np.mean(predictions[-10:])
        improvement = late_conf - early_conf
        score = max(0.0, min(1.0, improvement * 5))  # Scale to 0-1
    else:
        score = 0.0
    
    print(f"   ✅ Completed {cycles} cycles, score: {score:.3f}")
    return score

def quick_pattern_test(brain: BrainFactory, max_seconds: float = 5) -> float:
    """Quick pattern recognition test"""
    print("📊 Testing pattern recognition...")
    
    pattern_a = [0.8, 0.2] * 8  # High-low pattern
    pattern_b = [0.2, 0.8] * 8  # Low-high pattern
    
    start_time = time.time()
    responses_a = []
    responses_b = []
    cycles = 0
    
    while time.time() - start_time < max_seconds:
        # Alternate patterns
        if cycles % 2 == 0:
            action, _ = brain.process_sensory_input(pattern_a)
            responses_a.append(action[0] if action else 0)
        else:
            action, _ = brain.process_sensory_input(pattern_b)
            responses_b.append(action[0] if action else 0)
        
        cycles += 1
    
    # Score: different responses to different patterns
    if len(responses_a) > 5 and len(responses_b) > 5:
        mean_a = np.mean(responses_a[-5:])
        mean_b = np.mean(responses_b[-5:])
        difference = abs(mean_a - mean_b)
        score = min(1.0, difference * 10)  # Scale difference
    else:
        score = 0.0
    
    print(f"   ✅ Completed {cycles} cycles, score: {score:.3f}")
    return score

def quick_stability_test(brain: BrainFactory, max_seconds: float = 5) -> float:
    """Quick field stability test"""
    print("📊 Testing field stability...")
    
    stable_input = [0.5] * 16  # Constant input
    start_time = time.time()
    field_energies = []
    cycles = 0
    
    while time.time() - start_time < max_seconds:
        _, brain_state = brain.process_sensory_input(stable_input)
        
        # Track field energy
        energy = brain_state.get('field_energy', 0.0)
        field_energies.append(energy)
        
        cycles += 1
        if cycles % 10 == 0:
            print(f"   Cycle {cycles}: field energy = {energy:.3f}")
    
    # Score: decreasing and stabilizing energy
    if len(field_energies) > 20:
        early_energy = np.mean(field_energies[:10])
        late_energy = np.mean(field_energies[-10:])
        late_variance = np.var(field_energies[-10:])
        
        decrease_score = 1.0 if late_energy < early_energy else 0.5
        stability_score = max(0.0, 1.0 - late_variance * 10)
        score = (decrease_score + stability_score) / 2
    else:
        score = 0.0
    
    print(f"   ✅ Completed {cycles} cycles, score: {score:.3f}")
    return score

def main():
    """Run quick behavioral tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick behavioral test')
    parser.add_argument('--time-per-test', type=int, default=5,
                        help='Time per test in seconds (default: 5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    print(f"🧠 Quick Behavioral Test")
    print(f"⏱️  Time per test: {args.time_per_test} seconds")
    print("=" * 60)
    
    # Load configuration and create brain
    config = load_adaptive_configuration("settings.json")
    
    print("\n🔧 Creating brain...")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=args.quiet)
    
    # Run tests
    start_time = time.time()
    scores = {}
    
    print(f"\n🧪 Running behavioral tests...\n")
    
    scores['prediction'] = quick_prediction_test(brain, args.time_per_test)
    scores['pattern'] = quick_pattern_test(brain, args.time_per_test)
    scores['stability'] = quick_stability_test(brain, args.time_per_test)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    print(f"Prediction Learning: {scores['prediction']:.3f}")
    print(f"Pattern Recognition: {scores['pattern']:.3f}")
    print(f"Field Stability:     {scores['stability']:.3f}")
    print(f"Overall Score:       {np.mean(list(scores.values())):.3f}")
    print(f"\nTotal time: {total_time:.1f} seconds")
    
    # Cleanup
    if hasattr(brain, 'finalize_session'):
        brain.finalize_session()

if __name__ == "__main__":
    main()