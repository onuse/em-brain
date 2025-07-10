#!/usr/bin/env python3
"""
Analyze GUI performance bottlenecks systematically.
This script profiles the entire rendering pipeline to identify what's causing FPS degradation.
"""

import time
import sys
sys.path.append('.')

from monitoring.performance_profiler import RenderProfiler, profile_section
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from visualization.brain_monitor import BrainStateMonitor
from visualization.integrated_display import IntegratedDisplay
import pygame


def create_test_graph(num_nodes: int) -> WorldGraph:
    """Create a test graph with specified number of nodes."""
    print(f"Creating test graph with {num_nodes} nodes...")
    graph = WorldGraph()
    
    for i in range(num_nodes):
        experience = ExperienceNode(
            mental_context=[float(j % 10) for j in range(8)],
            action_taken={"forward_motor": 0.5, "turn_motor": 0.1, "brake_motor": 0.0},
            predicted_sensory=[0.4 + (i % 10) * 0.01] * 24,
            actual_sensory=[0.5 + (i % 10) * 0.01] * 24,
            prediction_error=0.1 + (i % 100) * 0.001,
            timestamp=time.time()
        )
        experience.strength = 1.0 + (i % 1000) / 100.0
        graph.add_node(experience)
    
    return graph


def profile_brain_monitor_update(monitor: BrainStateMonitor, graph: WorldGraph, profiler: RenderProfiler):
    """Profile a single brain monitor update cycle."""
    profiler.start_frame()
    
    with profile_section("brain_monitor_update"):
        # Simulate the update call
        with profile_section("prepare_data"):
            prediction_error = 0.1
            recent_action = {"forward_motor": 0.5, "turn_motor": 0.1, "brake_motor": 0.0}
            step = profiler.frame_count
        
        with profile_section("monitor.update"):
            monitor.update(graph, prediction_error, recent_action, step)
        
        # Profile the expensive parts separately
        with profile_section("get_graph_statistics"):
            stats = graph.get_graph_statistics()
        
        with profile_section("get_strongest_nodes"):
            strongest = graph.get_strongest_nodes(20)
        
        # Profile rendering components
        with profile_section("render_panels"):
            # Simulate rendering without actual pygame
            with profile_section("render_stats_panel"):
                pass  # Stats panel rendering
            
            with profile_section("render_node_strength_panel"):
                pass  # Node strength visualization
            
            with profile_section("render_prediction_graph"):
                pass  # Prediction error graph
            
            with profile_section("render_event_log"):
                pass  # Event log rendering
    
    profiler.end_frame(node_count=graph.node_count())


def analyze_scaling_behavior():
    """Analyze how performance scales with node count."""
    print("🔍 ANALYZING GUI PERFORMANCE SCALING")
    print("=" * 60)
    
    # Initialize profiler
    profiler = RenderProfiler()
    
    # Initialize pygame for brain monitor
    pygame.init()
    pygame.display.set_mode((800, 600))  # Dummy window
    
    # Test with different node counts
    node_counts = [0, 100, 500, 1000, 2000, 3000, 5000]
    
    for node_count in node_counts:
        print(f"\n📊 Testing with {node_count} nodes...")
        
        # Create test graph
        graph = create_test_graph(node_count)
        
        # Create brain monitor
        monitor = BrainStateMonitor()
        
        # Profile 100 frames
        frame_times = []
        for i in range(100):
            start = time.perf_counter()
            profile_brain_monitor_update(monitor, graph, profiler)
            end = time.perf_counter()
            frame_times.append(end - start)
        
        # Calculate statistics
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print(f"   Average frame time: {avg_frame_time*1000:.2f}ms")
        print(f"   Average FPS: {avg_fps:.1f}")
        
        # Get worst frame
        worst_frame_time = max(frame_times)
        worst_fps = 1.0 / worst_frame_time if worst_frame_time > 0 else 0
        print(f"   Worst frame: {worst_frame_time*1000:.2f}ms ({worst_fps:.1f} FPS)")
    
    # Save detailed report
    profiler.save_report("gui_scaling_analysis.json")
    
    # Additional analysis: profile specific operations
    print("\n🔬 DETAILED OPERATION PROFILING")
    print("-" * 40)
    
    # Test graph operations specifically
    test_graph = create_test_graph(2000)
    
    # Profile get_graph_statistics
    times = []
    for _ in range(100):
        start = time.perf_counter()
        stats = test_graph.get_graph_statistics()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"get_graph_statistics: {avg_time*1000:.2f}ms average")
    
    # Profile get_strongest_nodes
    times = []
    for _ in range(100):
        start = time.perf_counter()
        strongest = test_graph.get_strongest_nodes(20)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"get_strongest_nodes: {avg_time*1000:.2f}ms average")
    
    # Check if caching is working
    print("\n🔄 CACHE EFFECTIVENESS TEST")
    print("-" * 40)
    
    # First call (cache miss)
    start = time.perf_counter()
    stats1 = test_graph.get_graph_statistics()
    end = time.perf_counter()
    first_call_time = end - start
    
    # Second call (should be cached)
    start = time.perf_counter()
    stats2 = test_graph.get_graph_statistics()
    end = time.perf_counter()
    second_call_time = end - start
    
    print(f"First call (cache miss): {first_call_time*1000:.2f}ms")
    print(f"Second call (cached): {second_call_time*1000:.2f}ms")
    print(f"Cache speedup: {first_call_time/second_call_time:.1f}x")
    
    pygame.quit()


if __name__ == "__main__":
    analyze_scaling_behavior()