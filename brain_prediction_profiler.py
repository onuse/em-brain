#!/usr/bin/env python3
"""
Brain Prediction Pipeline Profiler
Detailed timing analysis to identify the real 200ms bottleneck
"""

import time
import functools
from typing import Dict, List, Any, Optional
from collections import defaultdict
from contextlib import contextmanager

class BrainPredictionProfiler:
    """Detailed profiler for brain prediction pipeline"""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.call_stack = []
        self.session_start = time.time()
        self.total_predictions = 0
        
    @contextmanager
    def time_section(self, section_name: str):
        """Context manager to time a code section"""
        start_time = time.time()
        self.call_stack.append(section_name)
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            full_name = " -> ".join(self.call_stack)
            self.timing_data[full_name].append(duration)
            self.call_stack.pop()
    
    def time_method(self, method_name: str = None):
        """Decorator to time method calls with call stack context"""
        def decorator(func):
            name = method_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.time_section(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_prediction_timing(self):
        """Start timing a complete brain prediction"""
        self.total_predictions += 1
        return self.time_section(f"prediction_{self.total_predictions}")
    
    def get_timing_analysis(self) -> Dict[str, Any]:
        """Get comprehensive timing analysis"""
        analysis = {
            'total_predictions': self.total_predictions,
            'session_duration': time.time() - self.session_start,
            'timing_breakdown': {},
            'top_bottlenecks': [],
            'method_stats': {}
        }
        
        # Calculate statistics for each timed section
        for section_name, times in self.timing_data.items():
            if times:
                stats = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'percentage_of_session': (sum(times) / analysis['session_duration']) * 100
                }
                analysis['timing_breakdown'][section_name] = stats
        
        # Find top bottlenecks by total time
        bottlenecks = [
            (section, stats['total_time'], stats['avg_time']) 
            for section, stats in analysis['timing_breakdown'].items()
        ]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)  # Sort by total time
        analysis['top_bottlenecks'] = bottlenecks[:10]
        
        # Calculate per-prediction average
        if self.total_predictions > 0:
            analysis['avg_prediction_time'] = analysis['session_duration'] / self.total_predictions
        
        return analysis
    
    def print_timing_report(self, top_n: int = 10):
        """Print a formatted timing report"""
        analysis = self.get_timing_analysis()
        
        print(f"\n🔍 BRAIN PREDICTION PROFILING REPORT")
        print("=" * 60)
        print(f"Total predictions: {analysis['total_predictions']}")
        print(f"Session duration: {analysis['session_duration']:.3f}s")
        
        if analysis['total_predictions'] > 0:
            print(f"Average prediction time: {analysis['avg_prediction_time']:.3f}s")
        
        print(f"\n⏱️ TOP {top_n} BOTTLENECKS (by total time):")
        print("-" * 60)
        
        for i, (section, total_time, avg_time) in enumerate(analysis['top_bottlenecks'][:top_n], 1):
            percentage = (total_time / analysis['session_duration']) * 100
            print(f"{i:2d}. {section}")
            print(f"    Total: {total_time:.3f}s | Avg: {avg_time:.6f}s | {percentage:.1f}% of session")
        
        print(f"\n📊 DETAILED BREAKDOWN:")
        print("-" * 60)
        
        # Group by top-level operations
        top_level_ops = defaultdict(float)
        for section, stats in analysis['timing_breakdown'].items():
            top_level = section.split(' -> ')[0]
            top_level_ops[top_level] += stats['total_time']
        
        for op, total_time in sorted(top_level_ops.items(), key=lambda x: x[1], reverse=True):
            percentage = (total_time / analysis['session_duration']) * 100
            print(f"  {op}: {total_time:.3f}s ({percentage:.1f}%)")
    
    def clear_data(self):
        """Clear all timing data"""
        self.timing_data.clear()
        self.call_stack.clear()
        self.total_predictions = 0
        self.session_start = time.time()


# Global profiler instance
_global_profiler: Optional[BrainPredictionProfiler] = None

def get_brain_profiler() -> BrainPredictionProfiler:
    """Get the global brain prediction profiler"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = BrainPredictionProfiler()
    return _global_profiler

def profile_method(method_name: str = None):
    """Decorator to profile a method using the global profiler"""
    return get_brain_profiler().time_method(method_name)

@contextmanager
def profile_section(section_name: str):
    """Context manager to profile a code section"""
    with get_brain_profiler().time_section(section_name):
        yield

if __name__ == "__main__":
    # Test the profiler
    profiler = BrainPredictionProfiler()
    
    # Simulate some operations
    with profiler.time_section("test_operation"):
        time.sleep(0.1)
        
        with profiler.time_section("sub_operation"):
            time.sleep(0.05)
    
    with profiler.time_section("another_operation"):
        time.sleep(0.02)
    
    profiler.print_timing_report()