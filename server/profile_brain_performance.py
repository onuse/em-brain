#!/usr/bin/env python3
"""
Brain Performance Profiler

Isolated test to identify cycle time bottlenecks using Python profiling tools.
"""

import sys
import os
import cProfile
import pstats
import time
import tempfile
from contextlib import contextmanager

# Add project root to path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brain import MinimalBrain


@contextmanager
def timer(description):
    """Context manager for timing code sections."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {(end - start) * 1000:.2f}ms")


def profile_brain_cycle_detailed():
    """Profile individual components of brain cycle."""
    print("🔬 Detailed Brain Cycle Profiling")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure for minimal overhead
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': True,
                'save_interval_cycles': 1000  # Reduce persistence overhead
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'target_cycle_time_ms': 0  # Disable sleep padding
            },
            'logging': {
                'log_brain_cycles': True,  # Keep logging to measure overhead
                'log_pattern_storage': False
            }
        }
        
        print("🧠 Creating brain...")
        with timer("Brain initialization"):
            brain = MinimalBrain(config=config, enable_logging=True, quiet_mode=True)
        
        # Warm up (exclude startup costs)
        print("\n🔥 Warming up (5 cycles)...")
        for i in range(5):
            sensory_input = [0.1 * i] * 16
            brain.process_sensory_input(sensory_input)
        
        print("\n📊 Profiling individual cycle components...")
        
        # Test sensory input
        sensory_input = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.7, 
                        0.6, 0.2, 0.8, 0.1, 0.5, 0.9, 0.3, 0.7]
        
        # Profile one complete cycle with detailed breakdown
        total_start = time.perf_counter()
        
        with timer("1. Input preprocessing"):
            # Simulate input preprocessing
            if len(sensory_input) > brain.sensory_dim:
                processed_input = sensory_input[:brain.sensory_dim]
            elif len(sensory_input) < brain.sensory_dim:
                processed_input = sensory_input + [0.0] * (brain.sensory_dim - len(sensory_input))
            else:
                processed_input = sensory_input
        
        with timer("2. Vector brain processing"):
            predicted_action, vector_brain_state = brain.vector_brain.process_sensory_input(processed_input)
        
        with timer("3. Confidence calculation"):
            confidence = vector_brain_state['prediction_confidence']
            prediction_error = 1.0 - confidence
        
        with timer("4. Cognitive autopilot update"):
            initial_brain_state = {
                'prediction_confidence': confidence,
                'prediction_error': prediction_error,
                'total_cycles': brain.total_cycles
            }
            autopilot_state = brain.cognitive_autopilot.update_cognitive_state(
                confidence, prediction_error, initial_brain_state
            )
        
        with timer("5. Hardware adaptation call"):
            from src.utils.hardware_adaptation import record_brain_cycle_performance
            record_brain_cycle_performance(10.0, 50.0)  # Simulate cycle time
        
        with timer("6. Brain state compilation"):
            brain_state = {
                'total_cycles': brain.total_cycles,
                'prediction_confidence': confidence,
                'prediction_error': prediction_error,
                'hardware_adaptive_limits': brain.hardware_adaptation.get_cognitive_limits(),
                'cognitive_autopilot': autopilot_state,
                'brain_uptime': time.time() - brain.brain_start_time,
                'architecture': 'sparse_goldilocks',
                **vector_brain_state
            }
        
        with timer("7. Logging operations"):
            if brain.logger:
                brain.logger.log_prediction_outcome(
                    predicted_action, sensory_input, confidence, 0
                )
        
        with timer("8. Persistence processing"):
            if brain.enable_persistence and brain.persistence_manager:
                brain.persistence_manager.process_brain_cycle(brain)
        
        total_time = (time.perf_counter() - total_start) * 1000
        print(f"\n🎯 Total cycle time: {total_time:.2f}ms")
        
        brain.finalize_session()


def profile_brain_loop_cprofile():
    """Profile brain loop using cProfile for hotspot identification."""
    print("\n🔬 cProfile Analysis (100 cycles)")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': True,
                'save_interval_cycles': 50
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'target_cycle_time_ms': 0
            },
            'logging': {
                'log_brain_cycles': True
            }
        }
        
        brain = MinimalBrain(config=config, enable_logging=True, quiet_mode=True)
        
        def run_brain_cycles():
            """Function to be profiled."""
            for i in range(100):
                sensory_input = [0.1 * (i % 10), 0.2 * (i % 7), 0.3 * (i % 5), 0.4 * (i % 3)] * 4
                brain.process_sensory_input(sensory_input)
        
        # Profile the function
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.perf_counter()
        run_brain_cycles()
        end_time = time.perf_counter()
        
        profiler.disable()
        brain.finalize_session()
        
        # Analyze results
        total_time = (end_time - start_time) * 1000
        avg_cycle_time = total_time / 100
        
        print(f"📊 100 cycles completed in {total_time:.1f}ms")
        print(f"📊 Average cycle time: {avg_cycle_time:.2f}ms")
        
        # Print top hotspots
        print(f"\n🔥 Top Performance Hotspots:")
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return avg_cycle_time


def profile_logging_overhead():
    """Profile just the logging overhead in isolation."""
    print("\n🔬 Logging Overhead Analysis")
    print("=" * 30)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with logging enabled
        config_with_logging = {
            'memory': {'persistent_memory_path': temp_dir, 'enable_persistence': False},
            'brain': {'type': 'sparse_goldilocks', 'target_cycle_time_ms': 0},
            'logging': {'log_brain_cycles': True}
        }
        
        brain_logged = MinimalBrain(config=config_with_logging, enable_logging=True, quiet_mode=True)
        
        # Warmup
        for i in range(5):
            brain_logged.process_sensory_input([0.1] * 16)
        
        # Time with logging
        start = time.perf_counter()
        for i in range(50):
            sensory_input = [0.1 * i] * 16
            brain_logged.process_sensory_input(sensory_input)
        logged_time = (time.perf_counter() - start) * 1000
        
        brain_logged.finalize_session()
        
        # Test with logging disabled
        config_no_logging = {
            'memory': {'persistent_memory_path': temp_dir, 'enable_persistence': False},
            'brain': {'type': 'sparse_goldilocks', 'target_cycle_time_ms': 0},
            'logging': {'log_brain_cycles': False}
        }
        
        brain_no_log = MinimalBrain(config=config_no_logging, enable_logging=False, quiet_mode=True)
        
        # Warmup
        for i in range(5):
            brain_no_log.process_sensory_input([0.1] * 16)
        
        # Time without logging
        start = time.perf_counter()
        for i in range(50):
            sensory_input = [0.1 * i] * 16
            brain_no_log.process_sensory_input(sensory_input)
        no_log_time = (time.perf_counter() - start) * 1000
        
        brain_no_log.finalize_session()
        
        # Results
        logged_avg = logged_time / 50
        no_log_avg = no_log_time / 50
        overhead = logged_avg - no_log_avg
        overhead_percent = (overhead / no_log_avg) * 100
        
        print(f"📊 With logging: {logged_avg:.2f}ms per cycle")
        print(f"📊 Without logging: {no_log_avg:.2f}ms per cycle")
        print(f"📊 Logging overhead: {overhead:.2f}ms ({overhead_percent:.1f}% increase)")
        
        return overhead


def main():
    """Run complete performance analysis."""
    print("🚀 Brain Performance Analysis Suite")
    print("=" * 60)
    
    try:
        # 1. Detailed component analysis
        profile_brain_cycle_detailed()
        
        # 2. Hotspot identification
        avg_cycle_time = profile_brain_loop_cprofile()
        
        # 3. Logging overhead measurement
        logging_overhead = profile_logging_overhead()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 Performance Analysis Summary:")
        print(f"   Average cycle time: {avg_cycle_time:.2f}ms")
        print(f"   Logging overhead: {logging_overhead:.2f}ms")
        print(f"   Core processing: {avg_cycle_time - logging_overhead:.2f}ms")
        
        if logging_overhead > 5.0:
            print("\n⚠️  Logging overhead is significant (>5ms)")
            print("   Recommendations:")
            print("   - Cache expensive method calls")
            print("   - Reduce logging frequency")
            print("   - Use object pooling for log objects")
        else:
            print("\n✅ Logging overhead is reasonable (<5ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)