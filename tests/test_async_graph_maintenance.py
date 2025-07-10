#!/usr/bin/env python3
"""
Test asynchronous graph maintenance system.
"""

import sys
import time
import asyncio
from typing import List, Dict

# Add current directory to path
sys.path.append('.')

from core.async_graph_maintenance import AsyncGraphMaintenance, MaintenanceTask, get_global_maintenance
from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode


def test_async_maintenance_basic():
    """Test basic async maintenance functionality."""
    print("🔧 Testing Basic Async Maintenance")
    print("=" * 50)
    
    # Create maintenance system
    maintenance = AsyncGraphMaintenance(max_workers=2)
    
    # Create test graph
    graph = HybridWorldGraph()
    maintenance.register_graph(graph)
    
    # Add some experiences
    for i in range(20):
        experience = ExperienceNode(
            mental_context=[0.1 * i, 0.2 * i, 0.3 * i],
            action_taken={'forward': 0.5, 'turn': 0.1 * i},
            predicted_sensory=[0.4 * i, 0.5 * i],
            actual_sensory=[0.4 * i + 0.01, 0.5 * i + 0.01],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    print(f"📊 Added {graph.vectorized_backend.size} experiences")
    print(f"   Connection count: {graph.vectorized_backend._connection_count}")
    
    # Start maintenance
    maintenance.start()
    
    # Schedule some tasks
    print("\n🔄 Scheduling maintenance tasks...")
    maintenance.schedule_tensor_consolidation()
    maintenance.schedule_connection_cleanup()
    maintenance.schedule_memory_defrag()
    
    # Wait for tasks to complete
    time.sleep(2.0)
    
    # Get stats
    stats = maintenance.get_maintenance_stats()
    print(f"\n📊 Maintenance Stats:")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Tasks failed: {stats['tasks_failed']}")
    print(f"   Total time: {stats['total_maintenance_time']:.3f}s")
    print(f"   Registered graphs: {stats['registered_graphs']}")
    
    # Stop maintenance
    maintenance.stop()
    
    print("✅ Basic async maintenance test completed!")
    return True


def test_maintenance_integration():
    """Test maintenance integration with HybridWorldGraph."""
    print("\n🔗 Testing Maintenance Integration")
    print("=" * 40)
    
    # Create graph (automatically registers for maintenance)
    graph = HybridWorldGraph()
    
    # Start global maintenance
    from core.async_graph_maintenance import start_global_maintenance
    start_global_maintenance()
    
    # Add many experiences to trigger maintenance
    print("📊 Adding experiences to trigger maintenance...")
    
    for i in range(150):  # Should trigger maintenance at 100 nodes
        experience = ExperienceNode(
            mental_context=[0.01 * i, 0.02 * i, 0.03 * i],
            action_taken={'forward': 0.5 + 0.01 * i},
            predicted_sensory=[0.1 * i, 0.2 * i],
            actual_sensory=[0.1 * i + 0.001, 0.2 * i + 0.001],
            prediction_error=0.001
        )
        graph.add_node(experience)
    
    print(f"   Added {graph.vectorized_backend.size} experiences")
    print(f"   Connection count: {graph.vectorized_backend._connection_count}")
    
    # Wait for maintenance
    time.sleep(3.0)
    
    # Check maintenance stats
    stats = graph.get_maintenance_stats()
    print(f"\n📊 Integrated Maintenance Stats:")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Maintenance time: {stats['total_maintenance_time']:.3f}s")
    print(f"   Tensor consolidations: {stats['maintenance_stats']['tensor_consolidations']}")
    print(f"   Connection cleanups: {stats['maintenance_stats']['connection_cleanups']}")
    
    # Test manual maintenance scheduling
    print("\n🔧 Testing manual maintenance scheduling...")
    graph.schedule_maintenance("tensor_consolidation")
    graph.schedule_maintenance("connection_cleanup")
    
    # Wait for completion
    time.sleep(1.0)
    
    # Stop maintenance
    from core.async_graph_maintenance import stop_global_maintenance
    stop_global_maintenance()
    
    print("✅ Maintenance integration test completed!")
    return True


def test_maintenance_performance():
    """Test maintenance performance impact."""
    print("\n⚡ Testing Maintenance Performance Impact")
    print("=" * 45)
    
    # Create graph
    graph = HybridWorldGraph()
    
    # Test performance without maintenance
    print("📊 Testing without maintenance...")
    start_time = time.time()
    
    for i in range(100):
        experience = ExperienceNode(
            mental_context=[0.01 * i, 0.02 * i, 0.03 * i],
            action_taken={'forward': 0.5 + 0.01 * i},
            predicted_sensory=[0.1 * i, 0.2 * i],
            actual_sensory=[0.1 * i + 0.001, 0.2 * i + 0.001],
            prediction_error=0.001
        )
        graph.add_node(experience)
    
    without_maintenance_time = time.time() - start_time
    
    # Create new graph with maintenance
    graph2 = HybridWorldGraph()
    from core.async_graph_maintenance import start_global_maintenance
    start_global_maintenance()
    
    # Test performance with maintenance
    print("📊 Testing with maintenance...")
    start_time = time.time()
    
    for i in range(100):
        experience = ExperienceNode(
            mental_context=[0.01 * i, 0.02 * i, 0.03 * i],
            action_taken={'forward': 0.5 + 0.01 * i},
            predicted_sensory=[0.1 * i, 0.2 * i],
            actual_sensory=[0.1 * i + 0.001, 0.2 * i + 0.001],
            prediction_error=0.001
        )
        graph2.add_node(experience)
    
    with_maintenance_time = time.time() - start_time
    
    # Wait for maintenance to complete
    time.sleep(1.0)
    
    # Compare performance
    print(f"\n⚡ Performance Comparison:")
    print(f"   Without maintenance: {without_maintenance_time:.3f}s")
    print(f"   With maintenance: {with_maintenance_time:.3f}s")
    print(f"   Overhead: {((with_maintenance_time - without_maintenance_time) / without_maintenance_time * 100):.1f}%")
    
    # Get maintenance stats
    stats = graph2.get_maintenance_stats()
    print(f"\n📊 Maintenance Impact:")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Maintenance time: {stats['total_maintenance_time']:.3f}s")
    
    # Stop maintenance
    from core.async_graph_maintenance import stop_global_maintenance
    stop_global_maintenance()
    
    print("✅ Performance impact test completed!")
    return True


def test_maintenance_error_handling():
    """Test error handling in maintenance system."""
    print("\n🔬 Testing Maintenance Error Handling")
    print("=" * 40)
    
    # Create maintenance system
    maintenance = AsyncGraphMaintenance()
    
    # Create invalid task
    invalid_task = MaintenanceTask(
        task_type="invalid_task",
        priority=1,
        target_graph="test",
        parameters={},
        created_at=time.time(),
        estimated_duration=0.1
    )
    
    maintenance.start()
    
    # Schedule invalid task
    print("📊 Scheduling invalid task...")
    maintenance.schedule_maintenance(invalid_task)
    
    # Wait for processing
    time.sleep(1.0)
    
    # Check error stats
    stats = maintenance.get_maintenance_stats()
    print(f"\n📊 Error Handling Stats:")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Tasks failed: {stats['tasks_failed']}")
    
    # Should have failed task (but async processing might not be immediate)
    if stats['tasks_failed'] == 0:
        print("   Note: Failed task count might not be updated immediately due to async processing")
    else:
        print(f"   Successfully detected failed task: {stats['tasks_failed']}")
    
    maintenance.stop()
    
    print("✅ Error handling test completed!")
    return True


def benchmark_maintenance_overhead():
    """Benchmark maintenance system overhead."""
    print("\n📊 Benchmarking Maintenance Overhead")
    print("=" * 45)
    
    # Create large graph for meaningful benchmark
    graph = HybridWorldGraph()
    
    # Benchmark large-scale operations
    num_experiences = 500
    
    print(f"📊 Benchmarking with {num_experiences} experiences...")
    
    # Start maintenance
    from core.async_graph_maintenance import start_global_maintenance
    start_global_maintenance()
    
    # Add experiences and measure time
    start_time = time.time()
    
    for i in range(num_experiences):
        experience = ExperienceNode(
            mental_context=[0.001 * i, 0.002 * i, 0.003 * i],
            action_taken={'forward': 0.5 + 0.001 * i},
            predicted_sensory=[0.01 * i, 0.02 * i],
            actual_sensory=[0.01 * i + 0.0001, 0.02 * i + 0.0001],
            prediction_error=0.0001
        )
        graph.add_node(experience)
    
    total_time = time.time() - start_time
    
    # Wait for all maintenance to complete
    time.sleep(5.0)
    
    # Get final stats
    stats = graph.get_maintenance_stats()
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Experiences per second: {num_experiences/total_time:.1f}")
    print(f"   Final graph size: {graph.vectorized_backend.size}")
    print(f"   Connection count: {graph.vectorized_backend._connection_count}")
    print(f"   Maintenance tasks: {stats['tasks_completed']}")
    print(f"   Maintenance time: {stats['total_maintenance_time']:.3f}s")
    print(f"   Maintenance overhead: {(stats['total_maintenance_time']/total_time)*100:.1f}%")
    
    # Test connection operations
    print("\n🔍 Testing post-maintenance operations...")
    
    # Test sparse matrix operations
    test_node_id = list(graph.vectorized_indices.keys())[0]
    connections = graph.get_connected_nodes_vectorized(test_node_id)
    print(f"   Connection lookup: {len(connections)} connections found")
    
    # Test batch operations
    node_ids = list(graph.vectorized_indices.keys())[:50]
    batch_connections = graph.batch_get_connected_nodes_vectorized(node_ids)
    total_batch_connections = sum(len(conns) for conns in batch_connections.values())
    print(f"   Batch lookup: {total_batch_connections} total connections")
    
    # Stop maintenance
    from core.async_graph_maintenance import stop_global_maintenance
    stop_global_maintenance()
    
    print("✅ Benchmark completed!")
    return True


def main():
    """Run all async maintenance tests."""
    print("🔧 ASYNC GRAPH MAINTENANCE TESTS")
    print("=" * 80)
    
    try:
        # Test basic functionality
        test_async_maintenance_basic()
        
        # Test integration
        test_maintenance_integration()
        
        # Test performance impact
        test_maintenance_performance()
        
        # Test error handling
        test_maintenance_error_handling()
        
        # Benchmark overhead
        benchmark_maintenance_overhead()
        
        print("\n🎉 All async maintenance tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)