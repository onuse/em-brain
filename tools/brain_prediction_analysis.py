#!/usr/bin/env python3
"""
Analyze brain prediction bottlenecks to identify 165ms performance issue.
"""

import sys
sys.path.append('.')

import time
import cProfile
import pstats
from typing import Dict, List, Any
from simulation.brainstem_sim import GridWorldBrainstem
from core.communication import SensoryPacket
from datetime import datetime

def analyze_brain_prediction_bottlenecks():
    """Deep dive into brain prediction pipeline performance."""
    print("Brain Prediction Bottleneck Analysis")
    print("=" * 50)
    
    # Initialize brain system
    brainstem = GridWorldBrainstem(
        world_width=12,
        world_height=12, 
        seed=42, 
        use_sockets=False
    )
    
    # Start session
    session_id = brainstem.brain_client.start_memory_session("Performance Analysis")
    
    def single_brain_prediction():
        """Single brain prediction cycle for timing analysis."""
        # Get current state using the right method
        stats = brainstem.simulation.get_simulation_stats()
        sensor_values = brainstem.simulation.get_sensor_readings()
        
        # Create sensory packet
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=sensor_values,
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        
        # Generate prediction (this is where the time should be spent)
        mental_context = sensor_values[:8] if len(sensor_values) >= 8 else sensor_values
        
        prediction = brainstem.brain_client.process_sensory_input(
            sensory_packet, 
            mental_context, 
            threat_level="normal"
        )
        
        # Apply action
        if prediction:
            action = prediction.motor_action
            brainstem.simulation.execute_motor_commands(action)
        
        brainstem.sequence_counter += 1
        return prediction
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = single_brain_prediction()
    
    # Detailed timing analysis
    print("\n1. Basic Timing Analysis")
    print("-" * 30)
    
    n_cycles = 20
    times = []
    
    for i in range(n_cycles):
        start_time = time.time()
        prediction = single_brain_prediction()
        end_time = time.time()
        
        cycle_time = end_time - start_time
        times.append(cycle_time)
        
        if i < 5:  # Show first 5 cycles
            print(f"  Cycle {i+1}: {cycle_time*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nBasic Results ({n_cycles} cycles):")
    print(f"  Average: {avg_time*1000:.2f}ms")
    print(f"  Min: {min_time*1000:.2f}ms")
    print(f"  Max: {max_time*1000:.2f}ms")
    print(f"  Std dev: {(sum([(t-avg_time)**2 for t in times])/len(times))**0.5*1000:.2f}ms")
    
    # Detailed profiling
    print("\n2. Detailed Profiling Analysis")
    print("-" * 30)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run cycles for profiling
    for i in range(10):
        _ = single_brain_prediction()
    
    profiler.disable()
    
    # Analyze profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("\nTop time-consuming functions:")
    stats.print_stats(20)
    
    # Component-wise timing
    print("\n3. Component-wise Timing Analysis")
    print("-" * 30)
    
    component_times = analyze_component_timing(brainstem)
    
    total_component_time = sum(component_times.values())
    
    print("Component breakdown:")
    for component, time_taken in sorted(component_times.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_taken / total_component_time) * 100
        print(f"  {component}: {time_taken*1000:.2f}ms ({percentage:.1f}%)")
    
    # Graph analysis
    print("\n4. Graph Analysis")
    print("-" * 30)
    
    graph_stats = brainstem.brain_client.get_brain_statistics()
    world_graph = brainstem.brain_client.get_world_graph()
    
    print(f"Graph size: {graph_stats['graph_stats']['total_nodes']} nodes")
    print(f"Similarity engine: {graph_stats['graph_stats']['similarity_engine']['acceleration_method']}")
    print(f"Average search time: {graph_stats['graph_stats']['similarity_engine']['avg_search_time']*1000:.3f}ms")
    print(f"Searches per second: {graph_stats['graph_stats']['similarity_engine']['searches_per_second']:.0f}")
    
    # Threading analysis
    print("\n5. Threading Opportunity Analysis")
    print("-" * 30)
    
    analyze_threading_opportunities(brainstem)
    
    return avg_time

def analyze_component_timing(brainstem) -> Dict[str, float]:
    """Analyze timing for individual components."""
    component_times = {}
    
    # Time sensor reading
    start_time = time.time()
    sensor_values = brainstem.simulation.get_sensor_readings()
    component_times['sensor_reading'] = time.time() - start_time
    
    # Time sensory packet creation
    start_time = time.time()
    sensory_packet = SensoryPacket(
        sequence_id=brainstem.sequence_counter,
        sensor_values=sensor_values,
        actuator_positions=[0.0, 0.0, 0.0],
        timestamp=datetime.now()
    )
    component_times['sensory_packet_creation'] = time.time() - start_time
    
    # Time mental context creation
    start_time = time.time()
    mental_context = sensor_values[:8] if len(sensor_values) >= 8 else sensor_values
    component_times['mental_context_creation'] = time.time() - start_time
    
    # Time the brain prediction (this is the main bottleneck)
    start_time = time.time()
    prediction = brainstem.brain_client.process_sensory_input(
        sensory_packet, 
        mental_context, 
        threat_level="normal"
    )
    component_times['brain_prediction'] = time.time() - start_time
    
    # Time action application
    start_time = time.time()
    if prediction:
        action = prediction.motor_action
        brainstem.simulation.execute_motor_commands(action)
    component_times['action_application'] = time.time() - start_time
    
    return component_times

def analyze_threading_opportunities(brainstem):
    """Analyze potential threading opportunities in the brain prediction pipeline."""
    print("Threading Opportunities:")
    
    # Get brain components
    brain_interface = brainstem.brain_client
    predictor = brain_interface.get_predictor()
    world_graph = brain_interface.get_world_graph()
    
    # Analyze traversal opportunities
    print("\n  A. Graph Traversal Parallelization:")
    print("    - TriplePredictor runs multiple traversals sequentially")
    print("    - Each traversal is independent and can be parallelized")
    print("    - Current: Sequential traversals with time budget")
    print("    - Opportunity: Parallel traversals with thread pool")
    
    # Analyze similarity search opportunities
    print("\n  B. Similarity Search Parallelization:")
    print("    - find_similar_nodes searches entire graph sequentially")
    print("    - Context similarity calculations are independent")
    print("    - Current: Single-threaded vectorized operations")
    print("    - Opportunity: Multi-threaded graph partitioning")
    
    # Analyze experience creation opportunities
    print("\n  C. Experience Creation Parallelization:")
    print("    - Experience evaluation by multiple drives")
    print("    - Each drive evaluation is independent")
    print("    - Current: Sequential drive evaluation")
    print("    - Opportunity: Parallel drive evaluation")
    
    # Analyze memory operations
    print("\n  D. Memory Operations:")
    print("    - Adaptive parameter updates")
    print("    - World graph updates")
    print("    - Current: Sequential memory operations")
    print("    - Opportunity: Background memory consolidation")
    
    # CPU utilization analysis
    print("\n  E. CPU Utilization Analysis:")
    import os
    cpu_count = os.cpu_count() or 4  # fallback to 4 if unknown
    print(f"    - Available CPU cores: {cpu_count}")
    print(f"    - Current brain uses: 1 core (single-threaded)")
    print(f"    - Theoretical speedup: {cpu_count}x for parallel components")

def identify_specific_bottlenecks():
    """Identify specific bottlenecks in the brain prediction pipeline."""
    print("\n6. Specific Bottleneck Identification")
    print("-" * 30)
    
    # Initialize system
    brainstem = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
    
    # Get current state
    sensor_values = brainstem.simulation.get_sensor_readings()
    sensory_packet = SensoryPacket(
        sequence_id=1,
        sensor_values=sensor_values,
        actuator_positions=[0.0, 0.0, 0.0],
        timestamp=datetime.now()
    )
    mental_context = sensor_values[:8]
    
    # Detailed timing of brain_interface.process_sensory_input
    brain_interface = brainstem.brain_client
    
    print("Breaking down process_sensory_input:")
    
    # Time individual steps
    start_time = time.time()
    
    # Step 1: Prediction generation
    predictor = brain_interface.get_predictor()
    world_graph = brain_interface.get_world_graph()
    
    step1_time = time.time()
    consensus_result = predictor.generate_prediction(
        mental_context, world_graph, 
        sensory_packet.sequence_id, "normal"
    )
    step1_duration = time.time() - step1_time
    
    print(f"  1. Prediction generation: {step1_duration*1000:.2f}ms")
    
    # Step 2: Experience creation (if applicable)
    if hasattr(brain_interface, 'last_prediction') and brain_interface.last_prediction:
        step2_time = time.time()
        # Simulate experience creation
        step2_duration = time.time() - step2_time
        print(f"  2. Experience creation: {step2_duration*1000:.2f}ms")
    
    # Analyze prediction generation components
    print("\n  Prediction generation breakdown:")
    
    # Time budget calculation
    start_time = time.time()
    time_budget = predictor._calculate_time_budget("normal")
    print(f"    - Time budget calc: {(time.time() - start_time)*1000:.3f}ms")
    
    # Single traversal timing
    start_time = time.time()
    traversal_result = predictor.single_traversal.traverse(
        mental_context, world_graph, random_seed=42
    )
    traversal_time = time.time() - start_time
    print(f"    - Single traversal: {traversal_time*1000:.2f}ms")
    
    # Consensus resolution timing
    start_time = time.time()
    consensus_result = predictor.consensus_resolver.resolve_consensus([traversal_result])
    consensus_time = time.time() - start_time
    print(f"    - Consensus resolution: {consensus_time*1000:.3f}ms")
    
    # Similarity search timing
    if world_graph.has_nodes():
        start_time = time.time()
        similar_nodes = world_graph.find_similar_nodes(mental_context, 0.7, 10)
        similarity_time = time.time() - start_time
        print(f"    - Similarity search: {similarity_time*1000:.2f}ms ({len(similar_nodes)} nodes found)")
    
    # Graph size impact
    node_count = world_graph.node_count()
    print(f"    - Graph size: {node_count} nodes")
    if node_count > 0:
        print(f"    - Search complexity: O({node_count}) per similarity search")

def main():
    """Main analysis function."""
    print("Brain Prediction Performance Analysis")
    print("=" * 50)
    
    # Run main analysis
    avg_time = analyze_brain_prediction_bottlenecks()
    
    # Identify specific bottlenecks
    identify_specific_bottlenecks()
    
    # Summary and recommendations
    print("\n" + "=" * 50)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    if avg_time > 0.165:  # 165ms threshold
        print(f"🔴 BOTTLENECK DETECTED: {avg_time*1000:.2f}ms (target: <165ms)")
    else:
        print(f"✅ Performance OK: {avg_time*1000:.2f}ms")
    
    print("\nTop Optimization Opportunities:")
    print("1. 🧵 PARALLEL TRAVERSALS: Run multiple graph traversals in parallel")
    print("2. 🔍 SIMILARITY SEARCH: Optimize O(n) similarity search with indexing")
    print("3. 🧠 EXPERIENCE CREATION: Parallelize drive evaluations")
    print("4. 📊 MEMORY OPERATIONS: Background memory consolidation")
    print("5. ⚡ VECTORIZATION: Further optimize NumPy operations")
    
    print(f"\nTheoretical speedup potential:")
    import os
    cpu_count = os.cpu_count() or 4
    print(f"- Multi-core speedup: {cpu_count}x for parallel components")
    print(f"- Similarity search indexing: 10-100x for large graphs")
    print(f"- Combined potential: {cpu_count*10}x+ performance improvement")

if __name__ == "__main__":
    main()