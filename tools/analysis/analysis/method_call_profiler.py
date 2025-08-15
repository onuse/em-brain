#!/usr/bin/env python3
"""
Method Call Profiler - Trace exactly which method call takes 2.37 seconds

This profiler instruments the MinimalBrain to time every method call
and identify the exact bottleneck.
"""

import time
import sys
import os
import signal
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
import torch

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

class MethodCallProfiler:
    """Profile individual method calls to find the exact bottleneck."""
    
    def __init__(self):
        self.call_times = {}
        self.call_stack = []
    
    def time_call(self, method_name, method_call):
        """Time a specific method call."""
        start_time = time.time()
        result = method_call()
        elapsed = (time.time() - start_time) * 1000
        
        self.call_times[method_name] = elapsed
        
        if elapsed > 100:  # Log slow calls immediately
            print(f"  🐌 SLOW: {method_name}: {elapsed:.1f}ms")
        elif elapsed > 10:
            print(f"  ⚠️  {method_name}: {elapsed:.1f}ms")
        else:
            print(f"  ✅ {method_name}: {elapsed:.2f}ms")
        
        return result
    
    def get_bottleneck_report(self):
        """Generate bottleneck report."""
        if not self.call_times:
            return "No timing data collected"
        
        # Sort by time
        sorted_calls = sorted(self.call_times.items(), key=lambda x: x[1], reverse=True)
        
        report = "\n📊 METHOD CALL TIMING BREAKDOWN\n"
        report += "=" * 50 + "\n"
        
        total_time = sum(self.call_times.values())
        
        for method_name, elapsed in sorted_calls:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            if elapsed > 1:  # Only show calls > 1ms
                report += f"  {method_name}: {elapsed:.1f}ms ({percentage:.1f}%)\n"
        
        report += f"\nTotal traced time: {total_time:.1f}ms\n"
        
        # Identify the bottleneck
        if sorted_calls:
            bottleneck_method, bottleneck_time = sorted_calls[0]
            report += f"\n🚨 PRIMARY BOTTLENECK: {bottleneck_method} ({bottleneck_time:.1f}ms)\n"
        
        return report

def trace_method_calls():
    """Trace method calls to find the 2.37s bottleneck."""
    print("🔍 METHOD CALL PROFILER")
    print("=" * 35)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    profiler = MethodCallProfiler()
    novel_input = [1.0, 2.0, 3.0, 4.0]
    
    print("\nTracing method calls in slow path...")
    
    try:
        with timeout(60):
            # Time the entire prediction first
            total_start = time.time()
            
            # Now manually trace each step in process_sensory_input
            print("\n1. Pre-processing...")
            
            # Time preprocessing
            def preprocess():
                if len(novel_input) > brain.sensory_dim:
                    return novel_input[:brain.sensory_dim]
                elif len(novel_input) < brain.sensory_dim:
                    return novel_input + [0.0] * (brain.sensory_dim - len(novel_input))
                else:
                    return novel_input
            
            processed_input = profiler.time_call("preprocess_sensory_input", preprocess)
            
            print("\n2. Vector brain processing...")
            
            # Time vector brain call
            def vector_brain_call():
                return brain.vector_brain.process_sensory_input(processed_input)
            
            predicted_action, vector_brain_state = profiler.time_call("vector_brain.process_sensory_input", vector_brain_call)
            
            print("\n3. Action dimension adjustment...")
            
            # Time action adjustment
            def adjust_action():
                action_dimensions = 2
                if action_dimensions and action_dimensions != len(predicted_action):
                    if action_dimensions < len(predicted_action):
                        return predicted_action[:action_dimensions]
                    else:
                        return predicted_action + [0.0] * (action_dimensions - len(predicted_action))
                return predicted_action
            
            final_action = profiler.time_call("adjust_action_dimensions", adjust_action)
            
            print("\n4. Cognitive autopilot update...")
            
            # Time cognitive autopilot
            def cognitive_autopilot_update():
                confidence = vector_brain_state['prediction_confidence']
                prediction_error = 1.0 - confidence
                initial_brain_state = {
                    'prediction_confidence': confidence,
                    'total_cycles': brain.total_cycles
                }
                return brain.cognitive_autopilot.update_cognitive_state(
                    confidence, prediction_error, initial_brain_state
                )
            
            autopilot_state = profiler.time_call("cognitive_autopilot.update_cognitive_state", cognitive_autopilot_update)
            
            print("\n5. Logging...")
            
            # Time logging
            def logging_call():
                if brain.logger and predicted_action:
                    brain.logger.log_prediction_outcome(
                        predicted_action, novel_input, vector_brain_state['prediction_confidence'], 0
                    )
                return True
            
            profiler.time_call("logger.log_prediction_outcome", logging_call)
            
            print("\n6. Performance recording...")
            
            # Time performance recording
            def performance_recording():
                from src.utils.hardware_adaptation import record_brain_cycle_performance
                cycle_time_ms = 100.0  # Dummy value
                memory_usage_mb = 50.0
                record_brain_cycle_performance(cycle_time_ms, memory_usage_mb)
                return True
            
            profiler.time_call("record_brain_cycle_performance", performance_recording)
            
            print("\n7. Brain state compilation...")
            
            # Time brain state compilation
            def brain_state_compilation():
                cycle_time = time.time() - total_start
                cycle_time_ms = cycle_time * 1000
                
                brain_state = {
                    'total_cycles': brain.total_cycles,
                    'prediction_confidence': vector_brain_state['prediction_confidence'],
                    'prediction_method': 'bootstrap_random' if brain.total_cycles == 0 else 'vector_stream_continuous',
                    'cycle_time': cycle_time,
                    'cycle_time_ms': cycle_time_ms,
                    'hardware_adaptive_limits': brain.hardware_adaptation.get_cognitive_limits(),
                    'cognitive_autopilot': autopilot_state,
                    'brain_uptime': time.time() - brain.brain_start_time,
                    'architecture': 'vector_stream',
                    **vector_brain_state
                }
                return brain_state
            
            brain_state = profiler.time_call("brain_state_compilation", brain_state_compilation)
            
            total_time = (time.time() - total_start) * 1000
            print(f"\nTotal manual trace time: {total_time:.1f}ms")
            
    except TimeoutError:
        print("❌ Method tracing timed out!")
        return
    except Exception as e:
        print(f"❌ Method tracing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print detailed report
    print(profiler.get_bottleneck_report())
    
    # Now test the actual full prediction for comparison
    print("\n🔄 FULL PREDICTION COMPARISON")
    print("=" * 40)
    
    try:
        with timeout(30):
            start_time = time.time()
            action, brain_state = brain.process_sensory_input(novel_input, action_dimensions=2)
            full_prediction_time = (time.time() - start_time) * 1000
            
            fast_path_used = brain_state.get('fast_path_used', False)
            
            print(f"Full prediction time: {full_prediction_time:.1f}ms")
            print(f"Fast path used: {fast_path_used}")
            print(f"Manual trace vs full: {abs(total_time - full_prediction_time):.1f}ms difference")
            
    except TimeoutError:
        print("❌ Full prediction comparison timed out!")
    
    return profiler

if __name__ == "__main__":
    print("Tracing method calls to find the 2.37s bottleneck...")
    profiler = trace_method_calls()
    
    if profiler:
        print(f"\n🏁 METHOD PROFILING COMPLETE")
        print(f"Check the timing breakdown above to find the exact bottleneck!")
    else:
        print(f"\n❌ METHOD PROFILING FAILED")