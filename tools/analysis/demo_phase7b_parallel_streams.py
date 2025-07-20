#!/usr/bin/env python3
"""
Demo: Phase 7b Parallel Stream Processing

Demonstrates the new parallel stream processing capabilities with 
biological timing coordination and performance comparison.
"""

import sys
import os
import time
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.brain_factory import MinimalBrain


def main():
    """Demo Phase 7b parallel stream processing."""
    print("🧠 Phase 7b Demo: Parallel Stream Processing")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration with parallel processing enabled
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': False
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'target_cycle_time_ms': 25.0,  # 40Hz gamma frequency
                'enable_biological_timing': True,
                'enable_parallel_processing': True  # Enable Phase 7b
            },
            'logging': {
                'log_brain_cycles': False
            }
        }
        
        print("🔧 Creating brain with parallel processing...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)
        
        print(f"\n🧬 Biological Timing Configuration:")
        if brain.biological_oscillator:
            timing = brain.biological_oscillator.get_current_timing()
            oscillator_stats = brain.biological_oscillator.get_oscillator_stats()
            print(f"  Gamma frequency: {1000.0 / brain.target_cycle_time_ms:.1f}Hz")
            print(f"  Target cycle time: {brain.target_cycle_time_ms}ms")
            print(f"  Current phase: {oscillator_stats['current_phase']}")
            print(f"  Binding window: {timing.binding_window_active}")
        
        # Demo: Compare sequential vs parallel processing
        print(f"\n📊 Performance Comparison Demo")
        print("-" * 30)
        
        # Test sequential mode
        print("🔄 Testing Sequential Mode (10 cycles)...")
        brain.enable_parallel_processing(False)
        
        sequential_start = time.time()
        for i in range(10):
            sensory_input = [0.1 * i, 0.2 * i, 0.3 * i] * 6  # Dynamic input
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            print(f"  Cycle {i+1}: {brain_state['cycle_time_ms']:.1f}ms (sequential)")
        sequential_time = time.time() - sequential_start
        
        # Test parallel mode
        print(f"\n⚡ Testing Parallel Mode (10 cycles)...")
        brain.enable_parallel_processing(True)
        
        parallel_start = time.time()
        for i in range(10):
            sensory_input = [0.1 * i, 0.2 * i, 0.3 * i] * 6  # Dynamic input
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            mode = "parallel" if brain_state.get('parallel_processing', False) else "sequential fallback"
            timing_info = ""
            if 'biological_timing' in brain_state:
                bio_timing = brain_state['biological_timing']
                timing_info = f" (γ:{bio_timing['gamma_phase']:.1f}°, θ:{bio_timing['theta_phase']:.1f}°)"
            
            print(f"  Cycle {i+1}: {brain_state['cycle_time_ms']:.1f}ms ({mode}){timing_info}")
        parallel_time = time.time() - parallel_start
        
        # Show performance stats
        performance_stats = brain.get_parallel_performance_stats()
        print(f"\n📈 Performance Results:")
        print(f"  Sequential total time: {sequential_time*1000:.1f}ms")
        print(f"  Parallel total time: {parallel_time*1000:.1f}ms")
        print(f"  Sequential cycles: {performance_stats.get('sequential_cycles', 0)}")
        print(f"  Parallel cycles: {performance_stats.get('parallel_cycles', 0)}")
        print(f"  Success rate: {performance_stats.get('parallel_success_rate', 0):.1%}")
        
        if 'speedup_ratio' in performance_stats:
            speedup = performance_stats['speedup_ratio']
            print(f"  Speedup: {speedup:.2f}x")
            
            if speedup > 1.0:
                print(f"🚀 Parallel processing achieved {speedup:.1f}x speedup!")
            else:
                print(f"📊 Parallel processing overhead: {(1/speedup - 1)*100:.1f}%")
        
        # Demo: Biological coordination
        print(f"\n🧬 Biological Coordination Demo")
        print("-" * 30)
        
        if brain.biological_oscillator:
            print("Monitoring biological timing over 5 cycles...")
            
            for i in range(5):
                sensory_input = [0.5, 0.3, 0.8] * 6
                motor_output, brain_state = brain.process_sensory_input(sensory_input)
                
                if 'biological_timing' in brain_state:
                    bio_timing = brain_state['biological_timing']
                    print(f"  Cycle {i+1}:")
                    print(f"    Gamma phase: {bio_timing['gamma_phase']:.1f}° ({bio_timing['current_phase']})")
                    print(f"    Theta phase: {bio_timing['theta_phase']:.1f}°")
                    print(f"    Binding window: {'✅' if bio_timing['binding_window_active'] else '❌'}")
                    print(f"    Consolidation: {'✅' if bio_timing['consolidation_active'] else '❌'}")
                    
                    if brain_state.get('parallel_processing', False):
                        cross_bindings = brain_state.get('cross_stream_bindings', 0)
                        print(f"    Cross-stream bindings: {cross_bindings}")
        
        # Demo: Stream coordination
        print(f"\n🔗 Stream Coordination Demo")
        print("-" * 30)
        
        brain.enable_parallel_processing(True)
        
        # Process a sequence that should create cross-stream bindings
        coordination_inputs = [
            [1.0, 0.0, 0.0] * 6,  # Strong sensory input
            [0.0, 1.0, 0.0] * 6,  # Different pattern
            [0.0, 0.0, 1.0] * 6,  # Another pattern
            [1.0, 1.0, 0.0] * 6,  # Combined pattern
            [0.5, 0.5, 0.5] * 6   # Balanced input
        ]
        
        print("Processing coordination sequence...")
        for i, sensory_input in enumerate(coordination_inputs):
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            print(f"  Step {i+1}: Confidence {brain_state['prediction_confidence']:.2f}")
            
            if brain_state.get('parallel_processing', False):
                bindings = brain_state.get('cross_stream_bindings', 0)
                coordination_success = brain_state.get('parallel_coordination_success', False)
                print(f"    Cross-stream bindings: {bindings}")
                print(f"    Coordination success: {'✅' if coordination_success else '❌'}")
        
        print(f"\n✅ Phase 7b Demo Complete!")
        print("Key achievements:")
        print("  🧬 Biological 40Hz gamma timing")
        print("  ⚡ Async parallel stream processing")
        print("  🔗 Cross-stream coordination")
        print("  📊 Performance monitoring")
        print("  🔄 Dynamic mode switching")
        
        brain.finalize_session()


if __name__ == "__main__":
    main()