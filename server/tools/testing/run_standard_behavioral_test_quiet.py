#!/usr/bin/env python3
"""
Standard 120-second behavioral test with enforced quiet mode
Prevents timeout issues from excessive logging
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import contextlib
import io

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

# Context manager to suppress stdout during noisy operations
@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

def run_behavioral_test(brain: BrainFactory, duration: float = 120):
    """Run comprehensive behavioral test with output suppression"""
    
    # Test 1: Prediction Learning
    print("\n📊 Test 1: Prediction Learning")
    print("-" * 40)
    
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    prediction_scores = []
    
    test_start = time.time()
    test_cycles = 0
    
    # Sample every 5 seconds
    sample_interval = 5.0
    next_sample = sample_interval
    
    while time.time() - test_start < duration / 3:  # 40 seconds for prediction
        # Suppress output during brain processing
        with suppress_stdout():
            action, brain_state = brain.process_sensory_input(pattern)
        
        confidence = brain_state.get('prediction_confidence', 0.0)
        test_cycles += 1
        
        # Sample periodically
        elapsed = time.time() - test_start
        if elapsed >= next_sample:
            prediction_scores.append(confidence)
            print(f"   {int(elapsed)}s: confidence = {confidence:.3f}")
            next_sample += sample_interval
    
    # Calculate prediction score
    if len(prediction_scores) >= 2:
        early = np.mean(prediction_scores[:2])
        late = np.mean(prediction_scores[-2:])
        prediction_score = min(1.0, max(0.0, (late - early) * 2))
    else:
        prediction_score = 0.0
    
    print(f"   Final: {test_cycles} cycles, score = {prediction_score:.3f}")
    
    # Test 2: Pattern Recognition
    print("\n📊 Test 2: Pattern Recognition")
    print("-" * 40)
    
    pattern_a = [0.8, 0.2] * 8
    pattern_b = [0.2, 0.8] * 8
    
    responses_a = []
    responses_b = []
    test_start = time.time()
    test_cycles = 0
    next_sample = sample_interval
    
    while time.time() - test_start < duration / 3:  # 40 seconds for patterns
        with suppress_stdout():
            if test_cycles % 2 == 0:
                action, _ = brain.process_sensory_input(pattern_a)
                responses_a.append(action[0])
            else:
                action, _ = brain.process_sensory_input(pattern_b)
                responses_b.append(action[0])
        
        test_cycles += 1
        
        # Sample periodically
        elapsed = time.time() - test_start
        if elapsed >= next_sample and len(responses_a) > 10 and len(responses_b) > 10:
            diff = abs(np.mean(responses_a[-10:]) - np.mean(responses_b[-10:]))
            print(f"   {int(elapsed)}s: pattern difference = {diff:.3f}")
            next_sample += sample_interval
    
    # Calculate pattern score
    if len(responses_a) > 20 and len(responses_b) > 20:
        final_diff = abs(np.mean(responses_a[-20:]) - np.mean(responses_b[-20:]))
        pattern_score = min(1.0, final_diff * 5)
    else:
        pattern_score = 0.0
    
    print(f"   Final: {test_cycles} cycles, score = {pattern_score:.3f}")
    
    # Test 3: Field Stability
    print("\n📊 Test 3: Field Stability")
    print("-" * 40)
    
    stable_input = [0.5] * 16
    field_energies = []
    test_start = time.time()
    test_cycles = 0
    next_sample = sample_interval
    
    while time.time() - test_start < duration / 3:  # 40 seconds for stability
        with suppress_stdout():
            _, brain_state = brain.process_sensory_input(stable_input)
        
        energy = brain_state.get('field_total_energy', 0.0)
        field_energies.append(energy)
        test_cycles += 1
        
        # Sample periodically
        elapsed = time.time() - test_start
        if elapsed >= next_sample:
            recent_energy = np.mean(field_energies[-50:]) if len(field_energies) > 50 else energy
            recent_var = np.var(field_energies[-50:]) if len(field_energies) > 50 else 0
            print(f"   {int(elapsed)}s: energy = {recent_energy:.1f}, variance = {recent_var:.1f}")
            next_sample += sample_interval
    
    # Calculate stability score
    if len(field_energies) > 100:
        early_var = np.var(field_energies[:50])
        late_var = np.var(field_energies[-50:])
        stability_improvement = max(0, (early_var - late_var) / max(early_var, 1))
        stability_score = min(1.0, stability_improvement)
    else:
        stability_score = 0.0
    
    print(f"   Final: {test_cycles} cycles, score = {stability_score:.3f}")
    
    return {
        'prediction': prediction_score,
        'pattern': pattern_score,
        'stability': stability_score,
        'overall': np.mean([prediction_score, pattern_score, stability_score])
    }

def main():
    """Run standard behavioral test with quiet mode"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standard behavioral test (quiet mode)')
    parser.add_argument('--duration', type=int, default=120,
                        help='Test duration in seconds (default: 120)')
    
    args = parser.parse_args()
    
    print(f"🧠 Standard Behavioral Test (Quiet Mode)")
    print(f"⏱️  Duration: {args.duration} seconds")
    print("=" * 60)
    
    # Load configuration and create brain
    config = load_adaptive_configuration("settings.json")
    
    print("\n🔧 Creating brain...")
    # Ensure quiet mode is enabled
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Run tests
    start_time = time.time()
    scores = run_behavioral_test(brain, args.duration)
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Prediction Learning: {scores['prediction']:.3f} {'✅' if scores['prediction'] >= 0.3 else '❌'}")
    print(f"Pattern Recognition: {scores['pattern']:.3f} {'✅' if scores['pattern'] >= 0.2 else '❌'}")
    print(f"Field Stability:     {scores['stability']:.3f} {'✅' if scores['stability'] >= 0.1 else '⚠️'}")
    print("-" * 60)
    print(f"Overall Score:       {scores['overall']:.3f}")
    print(f"\nTotal time: {total_time:.1f} seconds")
    
    # Interpretation
    print("\n📋 Assessment:")
    if scores['overall'] >= 0.5:
        print("✅ Excellent - Brain shows strong learning capabilities")
    elif scores['overall'] >= 0.3:
        print("✅ Good - Brain is learning effectively")
    elif scores['overall'] >= 0.2:
        print("⚠️  Fair - Brain shows basic learning")
    else:
        print("❌ Poor - Brain may need configuration adjustments")

if __name__ == "__main__":
    main()