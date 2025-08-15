#!/usr/bin/env python3
"""
Learning Speed Benchmark

Test the cumulative impact of all our optimizations on learning speed:
1. Baseline brain (no optimizations)
2. With biological laziness
3. With hardware-adaptive fuzzyness
4. With biologically realistic performance tiers
5. Combined optimizations

Measures:
- Experiences processed per second
- Average cycle time
- Memory efficiency
- Learning quality (prediction accuracy)
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple
from contextlib import contextmanager

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain
from biological_laziness_strategies import BiologicalLazinessManager
from biologically_realistic_fuzzyness import BiologicallyRealisticFuzzyness


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.time()
    yield lambda: (time.time() - start) * 1000
    

def generate_test_experiences(num_experiences: int = 200) -> List[Tuple[List[float], List[float]]]:
    """
    Generate realistic test experiences with patterns.
    
    Returns:
        List of (sensory_input, expected_outcome) pairs
    """
    experiences = []
    
    # Pattern 1: Linear mapping (20% of experiences)
    for i in range(int(num_experiences * 0.2)):
        sensory = [0.1 + i * 0.005, 0.2 + i * 0.005, 0.3 + i * 0.005, 0.4 + i * 0.005]
        expected = [s * 0.5 + 0.1 for s in sensory]
        experiences.append((sensory, expected))
    
    # Pattern 2: Repeated patterns with noise (40% of experiences) 
    patterns = [
        [0.3, 0.4, 0.5, 0.6],
        [0.6, 0.5, 0.4, 0.3],
        [0.2, 0.8, 0.2, 0.8]
    ]
    for i in range(int(num_experiences * 0.4)):
        base_pattern = patterns[i % len(patterns)]
        noise = np.random.normal(0, 0.02, 4)
        sensory = [max(0, min(1, b + n)) for b, n in zip(base_pattern, noise)]
        expected = [s * 0.8 + 0.1 for s in sensory]
        experiences.append((sensory, expected))
    
    # Pattern 3: Novel experiences (40% of experiences)
    for i in range(int(num_experiences * 0.4)):
        sensory = [np.random.random() for _ in range(4)]
        expected = [s * 0.7 + np.random.normal(0, 0.05) for s in sensory]
        experiences.append((sensory, expected))
    
    # Shuffle to create realistic mixed sequence
    np.random.shuffle(experiences)
    return experiences


def run_baseline_test(experiences: List[Tuple[List[float], List[float]]]) -> Dict:
    """Run baseline brain test without optimizations."""
    print("🔬 Running BASELINE test (no optimizations)")
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    cycle_times = []
    prediction_errors = []
    
    with timer() as get_time:
        for i, (sensory, expected_outcome) in enumerate(experiences):
            cycle_start = time.time()
            
            # Normal brain processing
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            simulated_outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, simulated_outcome, predicted_action)
            
            cycle_time = (time.time() - cycle_start) * 1000
            cycle_times.append(cycle_time)
            
            # Calculate prediction accuracy
            pred_error = np.mean([(p - e) ** 2 for p, e in zip(predicted_action, expected_outcome)])
            prediction_errors.append(pred_error)
    
    total_time = get_time()
    brain.finalize_session()
    
    return {
        'total_time': total_time,
        'avg_cycle_time': np.mean(cycle_times),
        'learning_rate': len(experiences) / (total_time / 1000),
        'avg_prediction_error': np.mean(prediction_errors),
        'experiences_processed': len(experiences),
        'optimizations_used': "None"
    }


def run_biological_laziness_test(experiences: List[Tuple[List[float], List[float]]]) -> Dict:
    """Run test with biological laziness optimizations."""
    print("🧠 Running BIOLOGICAL LAZINESS test")
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    laziness = BiologicalLazinessManager()
    
    cycle_times = []
    prediction_errors = []
    experiences_filtered = 0
    experiences_delayed = 0
    experiences_learned = 0
    
    with timer() as get_time:
        for i, (sensory, expected_outcome) in enumerate(experiences):
            cycle_start = time.time()
            
            # Get prediction first
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            confidence = brain_state.get('prediction_confidence', 0.5)
            
            # Calculate prediction error
            simulated_outcome = [a * 0.9 + 0.05 for a in predicted_action]
            prediction_error = np.mean([(p - o) ** 2 for p, o in zip(predicted_action, simulated_outcome)])
            
            # Apply biological laziness
            decision = laziness.should_process_experience(sensory, prediction_error, confidence)
            
            if decision['action'] == 'learn':
                brain.store_experience(sensory, predicted_action, simulated_outcome, predicted_action)
                experiences_learned += 1
                
                # Record learning outcome
                prediction_success = 1.0 - min(1.0, prediction_error)
                learn_time = (time.time() - cycle_start) * 1000
                laziness.record_learning_outcome(learn_time, prediction_success)
                
            elif decision['action'] == 'buffer':
                experiences_delayed += 1
            else:  # ignore
                experiences_filtered += 1
            
            # Update experience context
            laziness.add_experience_to_context(sensory, prediction_error)
            
            cycle_time = (time.time() - cycle_start) * 1000
            cycle_times.append(cycle_time)
            
            # Calculate prediction accuracy against expected
            pred_error = np.mean([(p - e) ** 2 for p, e in zip(predicted_action, expected_outcome)])
            prediction_errors.append(pred_error)
    
    total_time = get_time()
    brain.finalize_session()
    
    stats = laziness.get_statistics()
    
    return {
        'total_time': total_time,
        'avg_cycle_time': np.mean(cycle_times),
        'learning_rate': len(experiences) / (total_time / 1000),
        'avg_prediction_error': np.mean(prediction_errors),
        'experiences_processed': len(experiences),
        'experiences_learned': experiences_learned,
        'experiences_filtered': experiences_filtered,
        'filter_rate': stats['filter_rate'],
        'computational_savings': (stats['filter_rate'] + stats['delay_rate'] * 0.8) / 100,
        'optimizations_used': "Biological Laziness"
    }


def run_fuzzyness_test(experiences: List[Tuple[List[float], List[float]]]) -> Dict:
    """Run test with biologically realistic fuzzyness."""
    print("🎛️  Running FUZZYNESS test")
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    fuzzy_system = BiologicallyRealisticFuzzyness()
    
    cycle_times = []
    prediction_errors = []
    cache_hits = 0
    actual_predictions = 0
    patterns_skipped = 0
    
    with timer() as get_time:
        for i, (sensory, expected_outcome) in enumerate(experiences):
            cycle_start = time.time()
            
            # Check cache first
            cached_result = fuzzy_system.should_use_cache(sensory)
            
            if cached_result:
                predicted_action, brain_state = cached_result
                cache_hits += 1
            else:
                predicted_action, brain_state = brain.process_sensory_input(sensory)
                actual_predictions += 1
                fuzzy_system.cache_prediction(sensory, predicted_action, brain_state)
            
            # Simulate outcome
            simulated_outcome = [a * 0.9 + 0.05 for a in predicted_action]
            prediction_error = np.random.random() * 0.3  # Simulated error
            
            # Check if should filter
            if not fuzzy_system.should_filter_experience(sensory, prediction_error):
                # Store experience (maybe skip pattern analysis)
                if fuzzy_system.should_skip_pattern_analysis():
                    patterns_skipped += 1
                    # Minimal storage - just add to storage
                    from src.experience.models import Experience
                    exp = Experience(
                        sensory_input=sensory,
                        action_taken=predicted_action,
                        outcome=simulated_outcome,
                        prediction_error=prediction_error,
                        timestamp=time.time()
                    )
                    brain.experience_storage.add_experience(exp)
                else:
                    # Full storage
                    brain.store_experience(sensory, predicted_action, simulated_outcome, predicted_action)
            
            cycle_time = (time.time() - cycle_start) * 1000
            cycle_times.append(cycle_time)
            
            # Update performance tier
            fuzzy_system.update_performance_tier(cycle_time)
            
            # Calculate prediction accuracy
            pred_error = np.mean([(p - e) ** 2 for p, e in zip(predicted_action, expected_outcome)])
            prediction_errors.append(pred_error)
    
    total_time = get_time()
    brain.finalize_session()
    
    stats = fuzzy_system.get_statistics()
    
    return {
        'total_time': total_time,
        'avg_cycle_time': np.mean(cycle_times),
        'learning_rate': len(experiences) / (total_time / 1000),
        'avg_prediction_error': np.mean(prediction_errors),
        'experiences_processed': len(experiences),
        'cache_hits': cache_hits,
        'actual_predictions': actual_predictions,
        'cache_hit_rate': stats['cache_hit_rate'],
        'patterns_skipped': patterns_skipped,
        'final_tier': stats['current_tier'],
        'optimizations_used': "Biologically Realistic Fuzzyness"
    }


def run_combined_test(experiences: List[Tuple[List[float], List[float]]]) -> Dict:
    """Run test with both biological laziness AND fuzzyness."""
    print("🚀 Running COMBINED OPTIMIZATIONS test")
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    laziness = BiologicalLazinessManager()
    fuzzy_system = BiologicallyRealisticFuzzyness()
    
    cycle_times = []
    prediction_errors = []
    
    # Combined metrics
    cache_hits = 0
    actual_predictions = 0
    experiences_filtered = 0
    experiences_delayed = 0
    experiences_learned = 0
    patterns_skipped = 0
    
    with timer() as get_time:
        for i, (sensory, expected_outcome) in enumerate(experiences):
            cycle_start = time.time()
            
            # Check fuzzy cache first
            cached_result = fuzzy_system.should_use_cache(sensory)
            
            if cached_result:
                predicted_action, brain_state = cached_result
                cache_hits += 1
                confidence = brain_state.get('prediction_confidence', 0.8)  # Cached = high confidence
            else:
                predicted_action, brain_state = brain.process_sensory_input(sensory)
                actual_predictions += 1
                confidence = brain_state.get('prediction_confidence', 0.5)
                fuzzy_system.cache_prediction(sensory, predicted_action, brain_state)
            
            # Calculate prediction error
            simulated_outcome = [a * 0.9 + 0.05 for a in predicted_action]
            prediction_error = np.mean([(p - o) ** 2 for p, o in zip(predicted_action, simulated_outcome)])
            
            # Apply biological laziness decision
            decision = laziness.should_process_experience(sensory, prediction_error, confidence)
            
            if decision['action'] == 'learn':
                # Further check fuzzy filtering
                if not fuzzy_system.should_filter_experience(sensory, prediction_error):
                    # Decide on storage method
                    if fuzzy_system.should_skip_pattern_analysis():
                        patterns_skipped += 1
                        # Minimal storage
                        from src.experience.models import Experience
                        exp = Experience(
                            sensory_input=sensory,
                            action_taken=predicted_action,
                            outcome=simulated_outcome,
                            prediction_error=prediction_error,
                            timestamp=time.time()
                        )
                        brain.experience_storage.add_experience(exp)
                    else:
                        # Full storage
                        brain.store_experience(sensory, predicted_action, simulated_outcome, predicted_action)
                    
                    experiences_learned += 1
                    
                    # Record learning outcome for laziness
                    prediction_success = 1.0 - min(1.0, prediction_error)
                    learn_time = (time.time() - cycle_start) * 1000
                    laziness.record_learning_outcome(learn_time, prediction_success)
                
            elif decision['action'] == 'buffer':
                experiences_delayed += 1
            else:  # ignore
                experiences_filtered += 1
            
            # Update experience context for laziness
            laziness.add_experience_to_context(sensory, prediction_error)
            
            cycle_time = (time.time() - cycle_start) * 1000
            cycle_times.append(cycle_time)
            
            # Update fuzzy performance tier
            fuzzy_system.update_performance_tier(cycle_time)
            
            # Calculate prediction accuracy
            pred_error = np.mean([(p - e) ** 2 for p, e in zip(predicted_action, expected_outcome)])
            prediction_errors.append(pred_error)
    
    total_time = get_time()
    brain.finalize_session()
    
    laziness_stats = laziness.get_statistics()
    fuzzy_stats = fuzzy_system.get_statistics()
    
    return {
        'total_time': total_time,
        'avg_cycle_time': np.mean(cycle_times),
        'learning_rate': len(experiences) / (total_time / 1000),
        'avg_prediction_error': np.mean(prediction_errors),
        'experiences_processed': len(experiences),
        'experiences_learned': experiences_learned,
        'experiences_filtered': experiences_filtered,
        'cache_hits': cache_hits,
        'actual_predictions': actual_predictions,
        'cache_hit_rate': fuzzy_stats['cache_hit_rate'],
        'filter_rate': laziness_stats['filter_rate'],
        'patterns_skipped': patterns_skipped,
        'final_tier': fuzzy_stats['current_tier'],
        'computational_savings': (laziness_stats['filter_rate'] + laziness_stats['delay_rate'] * 0.8) / 100,
        'optimizations_used': "Biological Laziness + Fuzzyness"
    }


def benchmark_learning_speed():
    """Comprehensive learning speed benchmark."""
    print("🏁 LEARNING SPEED BENCHMARK")
    print("=" * 60)
    print("Testing cumulative impact of all optimizations on learning speed")
    print()
    
    # Generate consistent test data
    np.random.seed(42)  # For reproducible results
    experiences = generate_test_experiences(200)
    
    print(f"📊 Test data: {len(experiences)} experiences")
    print("   • 20% linear patterns")
    print("   • 40% repeated patterns with noise")
    print("   • 40% novel experiences")
    print()
    
    # Run all tests
    tests = [
        ("Baseline", run_baseline_test),
        ("Biological Laziness", run_biological_laziness_test),
        ("Fuzzyness", run_fuzzyness_test),
        ("Combined", run_combined_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func(experiences)
            results[test_name] = result
            
            print(f"✅ {test_name} completed:")
            print(f"   Total time: {result['total_time']:.0f}ms")
            print(f"   Learning rate: {result['learning_rate']:.1f} exp/sec")
            print(f"   Avg cycle: {result['avg_cycle_time']:.1f}ms")
            print(f"   Prediction error: {result['avg_prediction_error']:.4f}")
            
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            continue
    
    # Compare results
    print("\n" + "="*60)
    print("🎯 LEARNING SPEED COMPARISON")
    print("="*60)
    
    if "Baseline" in results:
        baseline = results["Baseline"]
        
        print(f"\n📈 Performance Improvements vs Baseline:")
        print("-" * 45)
        
        for test_name, result in results.items():
            if test_name == "Baseline":
                continue
            
            # Speed improvements
            speed_improvement = (result['learning_rate'] / baseline['learning_rate'] - 1) * 100
            time_reduction = (1 - result['total_time'] / baseline['total_time']) * 100
            cycle_improvement = (1 - result['avg_cycle_time'] / baseline['avg_cycle_time']) * 100
            
            print(f"\n{test_name}:")
            print(f"  Learning rate: {result['learning_rate']:.1f} exp/sec ({speed_improvement:+.1f}%)")
            print(f"  Total time: {result['total_time']:.0f}ms ({time_reduction:+.1f}%)")
            print(f"  Avg cycle: {result['avg_cycle_time']:.1f}ms ({cycle_improvement:+.1f}%)")
            
            # Accuracy impact
            accuracy_change = (result['avg_prediction_error'] / baseline['avg_prediction_error'] - 1) * 100
            print(f"  Prediction accuracy: {accuracy_change:+.1f}% error change")
            
            # Efficiency metrics
            if 'filter_rate' in result:
                print(f"  Computational savings: {result.get('computational_savings', 0)*100:.1f}%")
            if 'cache_hit_rate' in result:
                print(f"  Cache hit rate: {result['cache_hit_rate']:.1f}%")
            
            print(f"  Optimizations: {result['optimizations_used']}")
    
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS:")
    print("="*60)
    
    if "Combined" in results and "Baseline" in results:
        combined = results["Combined"]
        baseline = results["Baseline"]
        
        total_speedup = combined['learning_rate'] / baseline['learning_rate']
        total_savings = combined.get('computational_savings', 0) * 100
        
        print(f"\n🚀 Overall Impact of All Optimizations:")
        print(f"   Learning speed: {total_speedup:.2f}x faster")
        print(f"   Computational savings: {total_savings:.1f}%")
        print(f"   Cache efficiency: {combined.get('cache_hit_rate', 0):.1f}% hit rate")
        print(f"   Experience filtering: {combined.get('filter_rate', 0):.1f}% filtered")
        
        if total_speedup > 1.2:
            print(f"   ✅ SIGNIFICANT improvement - optimizations are working!")
        elif total_speedup > 1.05:
            print(f"   📊 MODERATE improvement - some benefit achieved")
        else:
            print(f"   ⚠️  MINIMAL improvement - optimizations may need refinement")


if __name__ == "__main__":
    benchmark_learning_speed()