#!/usr/bin/env python3
"""
Test realistic GUI performance including actual pygame rendering.
This will help identify if the bottleneck is in pygame rendering vs data processing.
"""

import time
import sys
sys.path.append('.')

import pygame
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from visualization.brain_monitor import BrainStateMonitor
from visualization.integrated_display import IntegratedDisplay
from simulation.brainstem_sim import GridWorldBrainstem


def create_test_graph(num_nodes: int) -> WorldGraph:
    """Create a test graph with specified number of nodes."""
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


def test_brain_monitor_rendering():
    """Test brain monitor rendering performance with real pygame."""
    print("🎮 TESTING BRAIN MONITOR RENDERING PERFORMANCE")
    print("=" * 60)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Brain Monitor Performance Test")
    
    # Create brain monitor
    monitor = BrainStateMonitor(400, 600)
    
    # Test with different node counts
    node_counts = [0, 100, 500, 1000, 2000, 3000, 5000]
    
    for node_count in node_counts:
        print(f"\n📊 Testing with {node_count} nodes...")
        
        # Create test graph
        graph = create_test_graph(node_count)
        
        # Warm up
        for _ in range(10):
            monitor.update(graph, 0.1, {'forward_motor': 0.5}, 0)
            monitor.render(screen, 100, 100)
            pygame.display.flip()
        
        # Measure rendering performance
        render_times = []
        update_times = []
        total_times = []
        
        for frame in range(100):
            # Time the update
            update_start = time.perf_counter()
            monitor.update(graph, 0.1, {'forward_motor': 0.5}, frame)
            update_end = time.perf_counter()
            update_times.append(update_end - update_start)
            
            # Time the render
            render_start = time.perf_counter()
            screen.fill((20, 20, 20))
            monitor.render(screen, 100, 100)
            pygame.display.flip()
            render_end = time.perf_counter()
            render_times.append(render_end - render_start)
            
            # Total frame time
            total_times.append((update_end - update_start) + (render_end - render_start))
            
            # Handle events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        # Calculate statistics
        avg_update = sum(update_times) / len(update_times)
        avg_render = sum(render_times) / len(render_times)
        avg_total = sum(total_times) / len(total_times)
        
        print(f"   Average update time: {avg_update*1000:.2f}ms")
        print(f"   Average render time: {avg_render*1000:.2f}ms")
        print(f"   Average total time: {avg_total*1000:.2f}ms")
        print(f"   Effective FPS: {1.0/avg_total:.1f}")
        
        # Check scaling
        if node_count == 0:
            baseline_render = avg_render
        else:
            slowdown = avg_render / baseline_render
            print(f"   Render slowdown vs 0 nodes: {slowdown:.1f}x")
    
    pygame.quit()


def test_integrated_display_performance():
    """Test the full integrated display performance."""
    print("\n\n🖥️  TESTING INTEGRATED DISPLAY PERFORMANCE")
    print("=" * 60)
    
    # Initialize brainstem
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    
    # Initialize display
    display = IntegratedDisplay(brainstem, cell_size=15)
    
    # Create test graphs
    test_graphs = {
        0: create_test_graph(0),
        500: create_test_graph(500),
        1000: create_test_graph(1000),
        2000: create_test_graph(2000),
        3000: create_test_graph(3000)
    }
    
    for node_count, graph in test_graphs.items():
        print(f"\n📊 Testing integrated display with {node_count} nodes...")
        
        # Set the brain graph
        display.set_brain_graph(graph)
        
        # Measure frame times
        frame_times = []
        clock = pygame.time.Clock()
        
        for frame in range(100):
            frame_start = time.perf_counter()
            
            # Render frame
            display._render_frame()
            
            # Control frame rate (simulate real usage)
            clock.tick(30)
            
            frame_end = time.perf_counter()
            frame_times.append(frame_end - frame_start)
            
            # Handle events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    display.cleanup()
                    return
        
        # Calculate statistics
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print(f"   Average frame time: {avg_frame_time*1000:.2f}ms")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Min FPS: {1.0/max(frame_times):.1f}")
        print(f"   Max FPS: {1.0/min(frame_times):.1f}")
    
    display.cleanup()


def identify_render_bottlenecks():
    """Profile individual rendering components."""
    print("\n\n🔍 IDENTIFYING SPECIFIC RENDER BOTTLENECKS")
    print("=" * 60)
    
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    
    # Create components
    monitor = BrainStateMonitor(400, 600)
    graph = create_test_graph(2000)
    
    # Profile each rendering section
    sections = [
        ('_draw_current_stats', lambda: monitor._draw_current_stats(10)),
        ('_draw_robot_mood', lambda: monitor._draw_robot_mood(100)),
        ('_draw_graph_visualization', lambda: monitor._draw_graph_visualization(200)),
        ('_draw_stats_graph', lambda: monitor._draw_stats_graph(350)),
        ('_draw_event_log', lambda: monitor._draw_event_log(450))
    ]
    
    # Update monitor with graph
    monitor.update(graph, 0.1, {'forward_motor': 0.5}, 0)
    
    print("\nProfiling individual render sections:")
    for section_name, section_func in sections:
        times = []
        
        for _ in range(100):
            start = time.perf_counter()
            try:
                section_func()
            except:
                pass  # Some sections might fail without full context
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f"   {section_name}: {avg_time*1000:.2f}ms average")
    
    # Test pygame.display.flip() performance
    flip_times = []
    for _ in range(100):
        start = time.perf_counter()
        pygame.display.flip()
        end = time.perf_counter()
        flip_times.append(end - start)
    
    avg_flip = sum(flip_times) / len(flip_times)
    print(f"\n   pygame.display.flip(): {avg_flip*1000:.2f}ms average")
    
    pygame.quit()


if __name__ == "__main__":
    print("🚀 COMPREHENSIVE GUI PERFORMANCE ANALYSIS")
    print("\nThis test measures real pygame rendering performance")
    print("to identify actual bottlenecks in the GUI.")
    print()
    
    test_brain_monitor_rendering()
    test_integrated_display_performance()
    identify_render_bottlenecks()
    
    print("\n✅ Performance analysis complete!")
    print("\nKey findings will help optimize the GUI for better FPS scaling.")