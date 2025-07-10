#!/usr/bin/env python3
"""
Comprehensive performance profiler for systematically tracing GUI bottlenecks.
Provides detailed timing breakdown of all rendering operations.
"""

import time
import functools
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from contextlib import contextmanager
import json
from pathlib import Path


class RenderProfiler:
    """Profile rendering pipeline to identify performance bottlenecks."""
    
    def __init__(self, output_dir: str = "./profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Timing data
        self.render_times = defaultdict(lambda: deque(maxlen=1000))
        self.call_stack = []
        self.frame_data = []
        
        # Frame tracking
        self.frame_count = 0
        self.session_start = time.perf_counter()
        
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling a code section."""
        start_time = time.perf_counter()
        self.call_stack.append((section_name, start_time))
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Record timing
            self.render_times[section_name].append(duration)
            
            # Build full call path
            call_path = "/".join([name for name, _ in self.call_stack])
            self.render_times[call_path].append(duration)
            
            self.call_stack.pop()
    
    def profile_method(self, method_name: str = None):
        """Decorator for profiling methods."""
        def decorator(func):
            name = method_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_section(name):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def start_frame(self):
        """Mark the start of a new frame."""
        self.current_frame_start = time.perf_counter()
        self.current_frame_sections = {}
        
    def end_frame(self, node_count: int = 0):
        """Mark the end of a frame and record data."""
        frame_end = time.perf_counter()
        frame_duration = frame_end - self.current_frame_start
        
        frame_info = {
            'frame': self.frame_count,
            'timestamp': frame_end - self.session_start,
            'duration': frame_duration,
            'fps': 1.0 / frame_duration if frame_duration > 0 else 0,
            'node_count': node_count,
            'sections': self._get_section_breakdown()
        }
        
        self.frame_data.append(frame_info)
        self.frame_count += 1
        
        # Print warning for slow frames
        if frame_duration > 0.033:  # Slower than 30 FPS
            self._print_slow_frame_warning(frame_info)
    
    def _get_section_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get timing breakdown for all sections in current frame."""
        breakdown = {}
        
        for section_name, timings in self.render_times.items():
            if timings:
                recent_timing = timings[-1]  # Most recent timing
                breakdown[section_name] = {
                    'time': recent_timing,
                    'percentage': 0  # Will be calculated later
                }
        
        # Calculate percentages
        total_time = sum(s['time'] for s in breakdown.values())
        if total_time > 0:
            for section in breakdown.values():
                section['percentage'] = (section['time'] / total_time) * 100
        
        return breakdown
    
    def _print_slow_frame_warning(self, frame_info: Dict):
        """Print warning for slow frames."""
        print(f"\n⚠️  SLOW FRAME DETECTED (Frame {frame_info['frame']})")
        print(f"   Total time: {frame_info['duration']*1000:.1f}ms ({frame_info['fps']:.1f} FPS)")
        print(f"   Node count: {frame_info['node_count']}")
        
        # Find top time consumers
        sections = frame_info['sections']
        top_sections = sorted(sections.items(), key=lambda x: x[1]['time'], reverse=True)[:5]
        
        print("   Top time consumers:")
        for section_name, data in top_sections:
            print(f"     {section_name}: {data['time']*1000:.1f}ms ({data['percentage']:.1f}%)")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.frame_data:
            return {}
        
        # Calculate statistics
        total_frames = len(self.frame_data)
        total_time = self.frame_data[-1]['timestamp'] if self.frame_data else 0
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        # FPS vs node count correlation
        fps_by_nodes = defaultdict(list)
        for frame in self.frame_data:
            node_bucket = (frame['node_count'] // 100) * 100  # Bucket by 100s
            fps_by_nodes[node_bucket].append(frame['fps'])
        
        fps_degradation = {}
        for nodes, fps_list in fps_by_nodes.items():
            fps_degradation[nodes] = {
                'avg_fps': sum(fps_list) / len(fps_list),
                'min_fps': min(fps_list),
                'max_fps': max(fps_list),
                'sample_count': len(fps_list)
            }
        
        # Section timing statistics
        section_stats = {}
        for section_name, timings in self.render_times.items():
            if timings and '/' not in section_name:  # Top-level sections only
                timing_list = list(timings)
                section_stats[section_name] = {
                    'avg_time': sum(timing_list) / len(timing_list),
                    'max_time': max(timing_list),
                    'min_time': min(timing_list),
                    'total_time': sum(timing_list),
                    'call_count': len(timing_list),
                    'time_per_frame': sum(timing_list) / total_frames
                }
        
        return {
            'summary': {
                'total_frames': total_frames,
                'total_time': total_time,
                'average_fps': avg_fps,
                'session_duration': time.perf_counter() - self.session_start
            },
            'fps_degradation': dict(sorted(fps_degradation.items())),
            'section_statistics': section_stats,
            'bottlenecks': self._identify_bottlenecks(section_stats)
        }
    
    def _identify_bottlenecks(self, section_stats: Dict) -> List[Dict]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Sort by total time consumed
        sorted_sections = sorted(
            section_stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        for section_name, stats in sorted_sections[:10]:
            bottleneck_info = {
                'section': section_name,
                'avg_time_ms': stats['avg_time'] * 1000,
                'max_time_ms': stats['max_time'] * 1000,
                'time_per_frame_ms': stats['time_per_frame'] * 1000,
                'total_percentage': (stats['total_time'] / sum(s['total_time'] for s in section_stats.values())) * 100
            }
            
            # Classify severity
            if bottleneck_info['avg_time_ms'] > 10:
                bottleneck_info['severity'] = 'critical'
            elif bottleneck_info['avg_time_ms'] > 5:
                bottleneck_info['severity'] = 'high'
            elif bottleneck_info['avg_time_ms'] > 2:
                bottleneck_info['severity'] = 'medium'
            else:
                bottleneck_info['severity'] = 'low'
            
            bottlenecks.append(bottleneck_info)
        
        return bottlenecks
    
    def save_report(self, filename: str = None):
        """Save performance report to file."""
        report = self.get_performance_report()
        
        if not filename:
            filename = f"performance_report_{int(time.time())}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Performance report saved to: {filepath}")
        self._print_report_summary(report)
    
    def _print_report_summary(self, report: Dict):
        """Print summary of performance report."""
        if not report:
            return
        
        summary = report['summary']
        print(f"\n🎯 PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total frames analyzed: {summary['total_frames']}")
        print(f"Average FPS: {summary['average_fps']:.1f}")
        
        print(f"\n📉 FPS Degradation by Node Count:")
        for nodes, stats in report['fps_degradation'].items():
            print(f"   {nodes:4d} nodes: {stats['avg_fps']:6.1f} FPS (min: {stats['min_fps']:.1f}, max: {stats['max_fps']:.1f})")
        
        print(f"\n🔥 Top Performance Bottlenecks:")
        for bottleneck in report['bottlenecks'][:5]:
            severity_emoji = {
                'critical': '🚨',
                'high': '⚠️ ',
                'medium': '⚡',
                'low': '📌'
            }.get(bottleneck['severity'], '❓')
            
            print(f"   {severity_emoji} {bottleneck['section']}: {bottleneck['avg_time_ms']:.1f}ms avg ({bottleneck['total_percentage']:.1f}% total)")


# Global profiler instance
_profiler: Optional[RenderProfiler] = None

def get_profiler() -> RenderProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = RenderProfiler()
    return _profiler

def profile_section(name: str):
    """Profile a code section."""
    return get_profiler().profile_section(name)

def profile_method(name: str = None):
    """Decorator to profile a method."""
    return get_profiler().profile_method(name)