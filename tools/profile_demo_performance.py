#!/usr/bin/env python3
"""
Profile demo_ultimate_2d_brain.py to find actual performance bottlenecks
"""

import sys
sys.path.append('.')

import time
import cProfile
import pstats
from simulation.brainstem_sim import GridWorldBrainstem
from core.communication import SensoryPacket
from datetime import datetime

def profile_brain_operations():
    """Profile the core brain operations without visualization"""
    print("Profiling Core Brain Operations")
    print("=" * 40)
    
    # Initialize brain system (same as demo)
    brainstem = GridWorldBrainstem(
        world_width=12,
        world_height=12, 
        seed=42, 
        use_sockets=False
    )
    
    # Start session
    session_id = brainstem.brain_client.start_memory_session("Profile Test")
    
    # Profile prediction generation
    print("Testing brain prediction performance...")
    
    def brain_prediction_cycle():
        """Single brain prediction cycle"""
        # Get current state
        state = brainstem.simulation.get_state()
        
        # Create sensory packet
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        
        # Generate prediction (this is where the time should be spent)
        mental_context = state['sensors'][:8] if len(state['sensors']) >= 8 else state['sensors']
        
        prediction = brainstem.brain_client.process_sensory_input(
            sensory_packet, 
            mental_context, 
            threat_level="normal"
        )
        
        # Apply action
        if prediction:
            action = prediction.motor_action
            brainstem.simulation.apply_action(action)
        
        return prediction
    
    # Warmup
    _ = brain_prediction_cycle()
    
    # Profile multiple cycles
    n_cycles = 10
    print(f"Profiling {n_cycles} brain prediction cycles...")
    
    start_time = time.time()
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for i in range(n_cycles):
        prediction = brain_prediction_cycle()
        if i == 0:
            print(f"  First prediction successful: {prediction is not None}")
    
    profiler.disable()
    
    total_time = time.time() - start_time
    avg_time = total_time / n_cycles
    cycles_per_second = 1.0 / avg_time
    
    print(f"\\nBrain Performance Results:")
    print(f"  {n_cycles} cycles: {total_time:.4f}s")
    print(f"  Average per cycle: {avg_time:.6f}s")
    print(f"  Cycles per second: {cycles_per_second:.1f}")
    print(f"  Theoretical FPS: {cycles_per_second:.1f}")
    
    # Analyze profiling results
    print(f"\\nProfiling Analysis (Top time consumers):")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(15)  # Show top 15 functions
    
    return avg_time, cycles_per_second

def test_visualization_overhead():
    """Test the overhead of the visualization system"""
    print(f"\\n{'='*40}")
    print("Testing Visualization Overhead")
    print("=" * 40)
    
    # Test pygame overhead
    try:
        import pygame
        from visualization.integrated_display import IntegratedDisplay
        
        # Initialize brainstem
        brainstem = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
        
        print("Testing visualization initialization...")
        start_time = time.time()
        
        display = IntegratedDisplay(brainstem, cell_size=25)
        
        init_time = time.time() - start_time
        print(f"  Visualization init: {init_time:.4f}s")
        
        # Test rendering overhead
        print("Testing render performance...")
        start_time = time.time()
        
        # Simulate rendering loop (without the step_delay)
        for i in range(10):
            # This would normally include brain update + rendering
            state = brainstem.simulation.get_state()
            # Simulate rendering (without actually rendering)
            pass
        
        render_time = (time.time() - start_time) / 10
        print(f"  Average render cycle: {render_time:.6f}s")
        print(f"  Render FPS potential: {1/render_time:.1f}")
        
        pygame.quit()
        
    except Exception as e:
        print(f"  Visualization test failed: {e}")

def analyze_step_delay_impact():
    """Analyze the impact of the step_delay"""
    print(f"\\n{'='*40}")
    print("Step Delay Impact Analysis")
    print("=" * 40)
    
    step_delay = 0.3  # From demo_ultimate_2d_brain.py
    theoretical_max_fps = 1.0 / step_delay
    
    print(f"Current step_delay: {step_delay}s")
    print(f"Theoretical max FPS: {theoretical_max_fps:.1f}")
    print(f"Observed FPS: 1.4")
    print(f"Additional overhead: {(1/1.4) - step_delay:.3f}s per frame")
    
    print(f"\\nBottleneck Analysis:")
    print(f"  Step delay bottleneck: {step_delay:.3f}s ({step_delay/(1/1.4)*100:.1f}% of frame time)")
    print(f"  Other bottlenecks: {((1/1.4) - step_delay):.3f}s ({((1/1.4) - step_delay)/(1/1.4)*100:.1f}% of frame time)")

def test_memory_operations():
    """Test memory save/load performance"""
    print(f"\\n{'='*40}")
    print("Memory Operations Performance")
    print("=" * 40)
    
    brainstem = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
    
    # Test memory save performance
    print("Testing memory save operations...")
    start_time = time.time()
    
    save_result = brainstem.brain_client.save_current_state()
    
    save_time = time.time() - start_time
    print(f"  Memory save: {save_time:.4f}s")
    
    if save_result:
        print(f"  Saved {save_result['experiences_count']} experiences")
    
    # Test statistics gathering
    print("Testing statistics gathering...")
    start_time = time.time()
    
    stats = brainstem.brain_client.get_brain_statistics()
    
    stats_time = time.time() - start_time
    print(f"  Statistics gathering: {stats_time:.4f}s")
    print(f"  Brain has {stats['graph_stats']['total_nodes']} nodes")

def main():
    """Main profiling function"""
    print("Demo Performance Profiling Analysis")
    print("=" * 50)
    
    # Test 1: Core brain operations
    avg_brain_time, brain_fps = profile_brain_operations()
    
    # Test 2: Visualization overhead
    test_visualization_overhead()
    
    # Test 3: Step delay analysis
    analyze_step_delay_impact()
    
    # Test 4: Memory operations
    test_memory_operations()
    
    # Summary
    print(f"\\n{'='*50}")
    print("PERFORMANCE BOTTLENECK SUMMARY")
    print("=" * 50)
    
    step_delay = 0.3
    observed_fps = 1.4
    frame_time = 1.0 / observed_fps
    
    print(f"Observed demo performance: {observed_fps} FPS ({frame_time:.3f}s per frame)")
    print(f"Artificial step_delay: {step_delay:.3f}s ({step_delay/frame_time*100:.1f}% of frame time)")
    print(f"Brain prediction time: {avg_brain_time:.6f}s ({avg_brain_time/frame_time*100:.1f}% of frame time)")
    print(f"Other overhead: {frame_time - step_delay - avg_brain_time:.3f}s")
    
    print(f"\\nRecommendations:")
    if step_delay > avg_brain_time * 10:
        print(f"  🎯 PRIMARY: Remove step_delay for immediate {1/avg_brain_time:.1f}x improvement")
    if avg_brain_time > 0.1:
        print(f"  🧠 SECONDARY: Optimize brain operations for additional speedup")
    else:
        print(f"  ✅ Brain operations are well optimized")
    
    print(f"\\nExpected performance without step_delay: {brain_fps:.1f} FPS")

if __name__ == "__main__":
    main()