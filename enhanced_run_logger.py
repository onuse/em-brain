#!/usr/bin/env python3
"""
Enhanced run logging with performance timing and method profiling
"""

import json
import time
import functools
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict, deque

class PerformanceTimer:
    """Simple performance timer for method calls"""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.active_timers = {}
        
    def time_method(self, method_name: str = None):
        """Decorator to time method calls"""
        def decorator(func):
            name = method_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    self.timing_data[name].append(duration)
                    
                    # Keep only last 100 measurements to avoid memory bloat
                    if len(self.timing_data[name]) > 100:
                        self.timing_data[name] = self.timing_data[name][-50:]
            
            return wrapper
        return decorator
    
    def start_timer(self, operation_name: str):
        """Start timing an operation"""
        self.active_timers[operation_name] = time.perf_counter()
    
    def end_timer(self, operation_name: str):
        """End timing an operation and record duration"""
        if operation_name in self.active_timers:
            start_time = self.active_timers.pop(operation_name)
            duration = time.perf_counter() - start_time
            self.timing_data[operation_name].append(duration)
            
            # Keep only last 100 measurements
            if len(self.timing_data[operation_name]) > 100:
                self.timing_data[operation_name] = self.timing_data[operation_name][-50:]
            
            return duration
        return None
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}
        
        for method_name, timings in self.timing_data.items():
            if timings:
                stats[method_name] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'recent_avg': sum(timings[-10:]) / min(10, len(timings))
                }
        
        return stats

class EnhancedRunLogger:
    """Enhanced run logger with performance timing"""
    
    def __init__(self, log_directory: str = "./run_logs"):
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True)
        
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("run_%Y%m%d_%H%M%S")
        
        # Data storage
        self.performance_data = []
        self.brain_snapshots = []
        self.events_log = []
        self.timing_samples = []
        
        # Performance timing
        self.timer = PerformanceTimer()
        self.frame_count = 0
        self.last_timing_report = time.time()
        
        print(f"📝 Enhanced Run Logger initialized: {self.session_id}")
    
    def log_frame_with_timing(self, brain_client, robot_state: Optional[Dict] = None):
        """Log a complete frame with timing breakdown"""
        frame_start = time.perf_counter()
        
        # Time brain statistics gathering
        self.timer.start_timer("get_brain_statistics")
        try:
            brain_stats = brain_client.get_brain_statistics()
        except Exception as e:
            brain_stats = {"error": str(e)}
        stats_duration = self.timer.end_timer("get_brain_statistics")
        
        # Time graph access
        self.timer.start_timer("get_world_graph")
        try:
            world_graph = brain_client.get_world_graph()
            nodes_count = len(world_graph.nodes) if world_graph else 0
        except Exception as e:
            nodes_count = 0
        graph_duration = self.timer.end_timer("get_world_graph")
        
        frame_end = time.perf_counter()
        total_frame_time = frame_end - frame_start
        fps = 1.0 / total_frame_time if total_frame_time > 0 else 0
        
        # Record timing sample
        timing_sample = {
            'frame': self.frame_count,
            'timestamp': frame_end,
            'total_frame_time': total_frame_time,
            'fps': fps,
            'brain_nodes': nodes_count,
            'timing_breakdown': {
                'brain_stats': stats_duration,
                'graph_access': graph_duration,
                'other': total_frame_time - (stats_duration or 0) - (graph_duration or 0)
            }
        }
        
        self.timing_samples.append(timing_sample)
        self.frame_count += 1
        
        # Log detailed performance every 50 frames
        if self.frame_count % 50 == 0:
            self.log_performance_summary()
        
        # Keep only recent samples to control memory
        if len(self.timing_samples) > 200:
            self.timing_samples = self.timing_samples[-100:]
        
        return timing_sample
    
    def log_performance_summary(self):
        """Log a performance summary"""
        current_time = time.time()
        
        if len(self.timing_samples) < 10:
            return
        
        # Get recent performance data
        recent_samples = self.timing_samples[-50:]
        
        avg_fps = sum(s['fps'] for s in recent_samples) / len(recent_samples)
        avg_frame_time = sum(s['total_frame_time'] for s in recent_samples) / len(recent_samples)
        
        # Get timing statistics
        timing_stats = self.timer.get_stats()
        
        performance_summary = {
            'timestamp': current_time,
            'frame_range': f"{self.frame_count-len(recent_samples)}-{self.frame_count}",
            'avg_fps': avg_fps,
            'avg_frame_time': avg_frame_time,
            'timing_stats': timing_stats,
            'top_time_consumers': self._get_top_time_consumers(timing_stats)
        }
        
        self.performance_data.append(performance_summary)
        
        # Print summary for immediate feedback
        if current_time - self.last_timing_report > 10:  # Every 10 seconds
            self._print_timing_summary(performance_summary)
            self.last_timing_report = current_time
    
    def _get_top_time_consumers(self, timing_stats: Dict) -> List[Dict]:
        """Get the methods taking the most time"""
        time_consumers = []
        
        for method, stats in timing_stats.items():
            total_time = stats['total_time']
            avg_time = stats['avg_time']
            time_consumers.append({
                'method': method,
                'total_time': total_time,
                'avg_time': avg_time,
                'call_count': stats['count']
            })
        
        # Sort by total time (highest first)
        time_consumers.sort(key=lambda x: x['total_time'], reverse=True)
        return time_consumers[:10]  # Top 10
    
    def _print_timing_summary(self, summary: Dict):
        """Print timing summary to console"""
        print(f"\\n⏱️  PERFORMANCE SUMMARY (Frame {summary['frame_range']})")
        print(f"   Average FPS: {summary['avg_fps']:.1f}")
        print(f"   Frame time: {summary['avg_frame_time']:.6f}s")
        
        if summary['top_time_consumers']:
            print("   Top time consumers:")
            for consumer in summary['top_time_consumers'][:5]:
                print(f"     {consumer['method']}: {consumer['avg_time']:.6f}s avg ({consumer['call_count']} calls)")
    
    def time_brain_prediction(self, prediction_func: Callable, *args, **kwargs):
        """Time a brain prediction call"""
        self.timer.start_timer("brain_prediction")
        try:
            result = prediction_func(*args, **kwargs)
            return result
        finally:
            duration = self.timer.end_timer("brain_prediction")
            # Removed redundant slow prediction logging - let performance summaries handle it
    
    def log_event(self, event_type: str, description: str, data: Optional[Dict] = None):
        """Log an event with timing"""
        timestamp = time.time()
        elapsed = timestamp - self.session_start.timestamp()
        
        event = {
            'timestamp': timestamp,
            'elapsed_seconds': elapsed,
            'frame': self.frame_count,
            'type': event_type,
            'description': description,
            'data': data or {}
        }
        
        self.events_log.append(event)
        
        # Print important events immediately
        if event_type in ['error', 'slow_prediction', 'performance_warning']:
            print(f"📋 {event_type.upper()}: {description}")
    
    def save_enhanced_log(self, brain_client, final_fps: float = 0.0):
        """Save comprehensive log with timing data"""
        try:
            # Get final timing statistics
            final_timing_stats = self.timer.get_stats()
            
            # Calculate session summary
            duration = time.time() - self.session_start.timestamp()
            total_frames = self.frame_count
            
            session_summary = {
                'session_id': self.session_id,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'total_frames': total_frames,
                'average_fps': total_frames / duration if duration > 0 else 0,
                'final_fps': final_fps,
                'timing_statistics': final_timing_stats
            }
            
            # Complete log data
            log_data = {
                'session_summary': session_summary,
                'performance_data': self.performance_data,
                'timing_samples': self.timing_samples[-100:],  # Last 100 samples
                'events_log': self.events_log,
                'brain_snapshots': self.brain_snapshots
            }
            
            # Save to file
            log_file = self.log_dir / f"{self.session_id}_enhanced.json"
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            print(f"💾 Enhanced log saved: {log_file}")
            self._print_final_summary(session_summary, final_timing_stats)
            
            return log_file
            
        except Exception as e:
            print(f"❌ Error saving enhanced log: {e}")
            return None
    
    def _print_final_summary(self, session_summary: Dict, timing_stats: Dict):
        """Print final session summary"""
        print(f"\\n📊 FINAL SESSION SUMMARY")
        print("=" * 50)
        print(f"Duration: {session_summary['duration_seconds']:.1f}s")
        print(f"Total frames: {session_summary['total_frames']}")
        print(f"Average FPS: {session_summary['average_fps']:.1f}")
        print(f"Final FPS: {session_summary['final_fps']:.1f}")
        
        if timing_stats:
            print(f"\\n⏱️  Performance Breakdown:")
            top_consumers = self._get_top_time_consumers(timing_stats)
            for consumer in top_consumers[:5]:
                percentage = (consumer['total_time'] / session_summary['duration_seconds']) * 100
                print(f"   {consumer['method']}: {percentage:.1f}% of total time")


# Global enhanced logger instance
_enhanced_logger: Optional[EnhancedRunLogger] = None

def get_enhanced_logger() -> EnhancedRunLogger:
    """Get the global enhanced logger instance"""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedRunLogger()
    return _enhanced_logger

def time_method(method_name: str = None):
    """Decorator to time method calls using the global logger"""
    return get_enhanced_logger().timer.time_method(method_name)

if __name__ == "__main__":
    print("📝 Enhanced Run Logger with Performance Timing")
    print("   Features:")
    print("   • Method call timing with decorators")
    print("   • Frame-by-frame performance breakdown")  
    print("   • Top time consumer identification")
    print("   • Automatic performance summaries")
    print("   • Minimal overhead design")