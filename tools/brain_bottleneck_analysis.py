#!/usr/bin/env python3
"""
Brain Bottleneck Analysis
Comprehensive profiling to identify the real 200ms prediction bottleneck
"""

import sys
sys.path.append('.')
import time
from simulation.brainstem_sim import GridWorldBrainstem
from core.communication import SensoryPacket
from datetime import datetime
from brain_prediction_profiler import get_brain_profiler

def run_bottleneck_analysis():
    """Run comprehensive bottleneck analysis"""
    print("🔍 BRAIN PREDICTION BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Initialize profiler
    profiler = get_brain_profiler()
    profiler.clear_data()
    
    print("Initializing brain system...")
    # Create brainstem with large brain (where we see the 200ms issue)
    brainstem = GridWorldBrainstem(4, 4, seed=42, use_sockets=False)
    
    # Get brain size
    brain_stats = brainstem.brain_client.get_brain_statistics()
    brain_size = brain_stats['graph_stats']['total_nodes']
    print(f"Brain size: {brain_size} experiences")
    
    # Run profiled predictions
    n_predictions = 10
    print(f"\nRunning {n_predictions} profiled predictions...")
    
    prediction_times = []
    
    for i in range(n_predictions):
        # Get sensor data
        sensor_values = brainstem.simulation.get_sensor_readings()
        sensory_packet = SensoryPacket(
            sequence_id=i,
            sensor_values=sensor_values,
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        mental_context = sensor_values[:8]
        
        # Time the complete prediction
        start_time = time.time()
        
        prediction = brainstem.brain_client.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        
        duration = time.time() - start_time
        prediction_times.append(duration)
        
        print(f"  Prediction {i+1}: {duration:.3f}s")
        
        # Add small delay to avoid overwhelming
        time.sleep(0.01)
    
    # Calculate overall statistics
    avg_time = sum(prediction_times) / len(prediction_times)
    theoretical_fps = 1.0 / avg_time
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Average prediction time: {avg_time:.3f}s")
    print(f"  Theoretical FPS: {theoretical_fps:.1f}")
    print(f"  Target (<165ms): {'✅ ACHIEVED' if avg_time < 0.165 else '❌ MISSED'}")
    
    # Print detailed profiling report
    print(f"\n" + "="*60)
    profiler.print_timing_report(top_n=15)
    
    # Get detailed analysis
    analysis = profiler.get_timing_analysis()
    
    print(f"\n🎯 BOTTLENECK IDENTIFICATION:")
    print("=" * 60)
    
    # Find the real culprits
    major_bottlenecks = []
    for section, total_time, avg_time in analysis['top_bottlenecks'][:10]:
        if avg_time > 0.01:  # More than 10ms average
            percentage = (total_time / analysis['session_duration']) * 100
            major_bottlenecks.append((section, avg_time, percentage))
    
    if major_bottlenecks:
        print("Major bottlenecks (>10ms average):")
        for i, (section, avg_time, percentage) in enumerate(major_bottlenecks, 1):
            print(f"  {i}. {section}")
            print(f"     Average: {avg_time:.3f}s ({percentage:.1f}% of total time)")
    else:
        print("No major bottlenecks found! All operations are fast.")
    
    # Analyze by operation type
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 60)
    
    for section, avg_time, percentage in major_bottlenecks[:3]:
        print(f"\n🎯 {section}:")
        print(f"   Current: {avg_time:.3f}s")
        
        if "traversals" in section.lower():
            print("   → Already optimized with parallel execution")
            print("   → Consider reducing traversal depth or count")
        elif "experience_creation" in section.lower():
            print("   → Consider async experience processing")
            print("   → Optimize drive evaluation loops")
        elif "drive" in section.lower():
            print("   → Consider parallel drive evaluation")
            print("   → Cache drive computations")
        elif "world_graph" in section.lower():
            print("   → Already has similarity acceleration")
            print("   → Consider background memory consolidation")
        elif "adaptive" in section.lower():
            print("   → Consider reducing adaptation frequency")
            print("   → Cache parameter calculations")
        else:
            print("   → Profile deeper to identify specific optimization")
    
    return analysis

if __name__ == "__main__":
    try:
        analysis = run_bottleneck_analysis()
        print(f"\n✅ Bottleneck analysis completed!")
        print(f"   Check the detailed report above for optimization targets.")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()