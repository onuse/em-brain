#!/usr/bin/env python3
"""
Performance Tracer for Brain Processing Pipeline

Let's trace exactly what's taking 2649ms and identify parallelization opportunities.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any
from contextlib import contextmanager

class PerformanceTracer:
    """Detailed performance tracing for brain processing."""
    
    def __init__(self):
        self.traces = []
        self.current_trace = None
        self.stack = []
    
    @contextmanager
    def trace(self, operation_name: str):
        """Context manager for tracing operation performance."""
        start_time = time.time()
        
        trace_entry = {
            'operation': operation_name,
            'start_time': start_time,
            'end_time': None,
            'duration_ms': None,
            'depth': len(self.stack),
            'children': []
        }
        
        # Add to parent's children if we have a parent
        if self.stack:
            self.stack[-1]['children'].append(trace_entry)
        else:
            self.traces.append(trace_entry)
        
        self.stack.append(trace_entry)
        
        try:
            yield trace_entry
        finally:
            end_time = time.time()
            trace_entry['end_time'] = end_time
            trace_entry['duration_ms'] = (end_time - start_time) * 1000
            self.stack.pop()
    
    def get_performance_report(self) -> str:
        """Generate detailed performance report."""
        report = "🔍 PERFORMANCE TRACE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for trace in self.traces:
            report += self._format_trace(trace, 0)
        
        return report
    
    def _format_trace(self, trace: Dict[str, Any], depth: int) -> str:
        """Format a single trace entry."""
        indent = "  " * depth
        duration = trace['duration_ms']
        
        if duration > 1000:
            duration_str = f"{duration:.0f}ms"
        elif duration > 100:
            duration_str = f"{duration:.1f}ms"
        else:
            duration_str = f"{duration:.2f}ms"
        
        result = f"{indent}📊 {trace['operation']}: {duration_str}\n"
        
        # Add children
        for child in trace['children']:
            result += self._format_trace(child, depth + 1)
        
        return result
    
    def get_bottlenecks(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """Get operations that took longer than threshold."""
        bottlenecks = []
        
        def find_bottlenecks(trace):
            if trace['duration_ms'] > threshold_ms:
                bottlenecks.append({
                    'operation': trace['operation'],
                    'duration_ms': trace['duration_ms'],
                    'depth': trace['depth']
                })
            
            for child in trace['children']:
                find_bottlenecks(child)
        
        for trace in self.traces:
            find_bottlenecks(trace)
        
        return sorted(bottlenecks, key=lambda x: x['duration_ms'], reverse=True)
    
    def reset(self):
        """Reset tracer for new measurement."""
        self.traces = []
        self.current_trace = None
        self.stack = []


def trace_brain_processing():
    """Trace the brain processing pipeline to identify bottlenecks."""
    print("🔍 TRACING BRAIN PROCESSING PIPELINE")
    print("=" * 50)
    
    # Import after path setup
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))
    
    from vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
    
    # Create tracer
    tracer = PerformanceTracer()
    
    # Create brain
    print("Creating brain...")
    brain = SparseGoldilocksBrain(quiet_mode=True)
    
    # Trace single prediction cycle
    print("Tracing single prediction cycle...")
    
    with tracer.trace("TOTAL_PREDICTION_CYCLE"):
        sensory_input = [1.0, 2.0, 3.0, 4.0]
        
        with tracer.trace("BRAIN_PROCESS_SENSORY_INPUT"):
            action, brain_state = brain.process_sensory_input(sensory_input, action_dimensions=2)
    
    # Generate report
    print(tracer.get_performance_report())
    
    # Identify bottlenecks
    bottlenecks = tracer.get_bottlenecks(threshold_ms=50)
    
    print("\n🚨 PERFORMANCE BOTTLENECKS (>50ms):")
    print("-" * 40)
    for bottleneck in bottlenecks:
        print(f"  {bottleneck['operation']}: {bottleneck['duration_ms']:.1f}ms")
    
    return tracer, bottlenecks


def analyze_parallelization_opportunities():
    """Analyze what operations could be parallelized."""
    print("\n🔄 PARALLELIZATION ANALYSIS")
    print("=" * 50)
    
    analysis = {
        'sequential_operations': [
            'tensor_conversion',
            'sparse_encoding', 
            'temporal_hierarchy_processing',
            'competitive_dynamics',
            'brain_state_compilation'
        ],
        'parallel_opportunities': [
            'stream_updates (sensory, motor, temporal)',
            'pattern_similarity_searches',
            'cross_stream_coactivation',
            'multiple_budget_predictions'
        ],
        'current_bottlenecks': [
            'emergent_hierarchy.process_with_adaptive_budget()',
            'cortical_column_clustering',
            'dense_pattern_conversions',
            'pattern_search_iterations'
        ]
    }
    
    print("📋 Current Sequential Operations:")
    for op in analysis['sequential_operations']:
        print(f"  • {op}")
    
    print("\n⚡ Parallelization Opportunities:")
    for op in analysis['parallel_opportunities']:
        print(f"  • {op}")
    
    print("\n🚨 Current Bottlenecks:")
    for op in analysis['current_bottlenecks']:
        print(f"  • {op}")
    
    return analysis


if __name__ == "__main__":
    # Trace brain processing
    tracer, bottlenecks = trace_brain_processing()
    
    # Analyze parallelization
    analysis = analyze_parallelization_opportunities()
    
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    print(f"Total bottlenecks found: {len(bottlenecks)}")
    print(f"Parallelization opportunities: {len(analysis['parallel_opportunities'])}")
    print(f"Current architecture: Sequential pipeline")
    print(f"Recommendation: Focus on top 3 bottlenecks for maximum impact")