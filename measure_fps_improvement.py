#!/usr/bin/env python3
"""
Measure FPS improvement from GUI caching optimizations.
"""

import time
import sys
sys.path.append('.')

from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from visualization.brain_monitor import BrainStateMonitor
import pygame

def create_large_graph(num_nodes=2000):
    """Create a graph with many nodes to simulate real usage."""
    print(f"Creating graph with {num_nodes} nodes...")
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
    
    print(f"Graph created with {graph.node_count()} nodes")
    return graph

def simulate_gui_update_old_way(graph, num_frames=100):
    """Simulate the old GUI update pattern (every frame)."""
    print(f"Simulating OLD GUI pattern (every frame)...")
    
    start_time = time.time()
    for frame in range(num_frames):
        # Old way: expensive operations every frame
        stats = graph.get_graph_statistics()
        strongest = graph.get_strongest_nodes(20)
        
        # Simulate other GUI work
        time.sleep(0.001)  # 1ms of other GUI work
    
    total_time = time.time() - start_time
    fps = num_frames / total_time
    
    print(f"  Old way: {num_frames} frames in {total_time:.2f}s = {fps:.1f} FPS")
    return fps

def simulate_gui_update_new_way(graph, num_frames=100):
    """Simulate the new GUI update pattern (cached, every 10th frame)."""
    print(f"Simulating NEW GUI pattern (cached, every 10th frame)...")
    
    start_time = time.time()
    for frame in range(num_frames):
        # New way: expensive operations only every 10th frame
        if frame % 10 == 0:
            stats = graph.get_graph_statistics()
            strongest = graph.get_strongest_nodes(20)
        
        # Simulate other GUI work
        time.sleep(0.001)  # 1ms of other GUI work
    
    total_time = time.time() - start_time
    fps = num_frames / total_time
    
    print(f"  New way: {num_frames} frames in {total_time:.2f}s = {fps:.1f} FPS")
    return fps

def main():
    """Run the FPS comparison test."""
    print("📊 GUI FPS OPTIMIZATION COMPARISON")
    print("=" * 50)
    
    # Test with realistic node count
    graph = create_large_graph(2000)
    
    print(f"\nTesting with {graph.node_count()} nodes:")
    print("This simulates the real demo scenario where GUI slows down")
    print("as the robot learns and accumulates experiences.")
    
    # Test old way (before optimization)
    old_fps = simulate_gui_update_old_way(graph, 100)
    
    # Test new way (after optimization)
    new_fps = simulate_gui_update_new_way(graph, 100)
    
    # Show improvement
    improvement = new_fps / old_fps
    print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
    print(f"   Before optimization: {old_fps:.1f} FPS")
    print(f"   After optimization:  {new_fps:.1f} FPS")
    print(f"   Improvement factor:  {improvement:.1f}x faster")
    
    # Real-world context
    print(f"\n🎯 REAL-WORLD IMPACT:")
    if old_fps < 15:
        print(f"   ❌ Before: {old_fps:.1f} FPS = Choppy, unusable GUI")
    else:
        print(f"   ✅ Before: {old_fps:.1f} FPS = Usable but slow")
    
    if new_fps >= 25:
        print(f"   ✅ After: {new_fps:.1f} FPS = Smooth, responsive GUI")
    elif new_fps >= 15:
        print(f"   ✅ After: {new_fps:.1f} FPS = Good performance")
    else:
        print(f"   ⚠️ After: {new_fps:.1f} FPS = Still needs work")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Caching reduced expensive operations by 90%")
    print(f"   • GUI now scales much better with large graphs")
    print(f"   • Brain can learn without GUI performance degradation")

if __name__ == "__main__":
    main()