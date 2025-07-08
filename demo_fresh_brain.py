#!/usr/bin/env python3
"""
Fresh brain demo - starts with empty brain for maximum performance testing
"""

import sys
sys.path.append('.')

import time
import pygame
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.integrated_display import IntegratedDisplay


def main():
    """Launch fresh brain demo with no persistence."""
    print("🆕 FRESH BRAIN PERFORMANCE TEST")
    print("=" * 50)
    print("Starting with completely fresh brain:")
    print("• No persistent memory")
    print("• No saved experiences")
    print("• Maximum performance")
    print()
    
    # Create brainstem but avoid loading any persistent data
    brainstem = GridWorldBrainstem(
        world_width=12,
        world_height=12, 
        seed=42, 
        use_sockets=False
    )
    
    # Skip starting memory session to avoid persistence
    print("🧠 Fresh brain initialized...")
    print("   No memory loading - starting from scratch")
    print("   This should show maximum brain performance")
    print()
    
    # Initialize visualization
    display = IntegratedDisplay(brainstem, cell_size=25)
    
    # Simple brain callback that doesn't trigger persistence
    def fresh_brain_callback(state):
        """Simple brain prediction for fresh brain testing."""
        from core.communication import SensoryPacket
        from datetime import datetime
        
        # Create minimal sensory packet
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        
        # Use simple context
        mental_context = state['sensors'][:8] if len(state['sensors']) >= 8 else state['sensors']
        
        # Get prediction (should be very fast with empty brain)
        prediction = brainstem.brain_client.process_sensory_input(
            sensory_packet, 
            mental_context, 
            threat_level="normal"
        )
        
        return prediction.motor_action if prediction else {
            'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.0
        }
    
    display.set_learning_callback(fresh_brain_callback)
    
    # Connect fresh brain graph
    brain_graph = brainstem.brain_client.get_world_graph()
    display.set_brain_graph(brain_graph)
    
    try:
        print("🚀 Launching fresh brain test...")
        print("   Expected: Very high FPS with empty brain")
        print("   The brain will learn in real-time but not save")
        print()
        
        # Measure performance
        start_time = time.time()
        
        # Run without delays
        display.run(
            auto_step=True,
            step_delay=0.0  # Maximum speed
        )
        
    except KeyboardInterrupt:
        print("\\n⏹️  Fresh brain test interrupted")
        
        # Quick performance summary
        elapsed = time.time() - start_time
        print(f"   Test duration: {elapsed:.1f} seconds")
        
        # Check brain growth
        stats = brainstem.brain_client.get_brain_statistics()
        nodes = stats.get('graph_stats', {}).get('total_nodes', 0)
        print(f"   Brain grew to: {nodes} experience nodes")
        
        if elapsed > 5:  # Only calculate if ran for reasonable time
            estimated_fps = nodes / elapsed if elapsed > 0 else 0
            print(f"   Estimated node creation rate: {estimated_fps:.1f} nodes/second")
        
    except Exception as e:
        print(f"\\n❌ Fresh brain test error: {e}")
    finally:
        print("\\n🏁 Fresh brain test completed")
        print("   No data saved - this was a performance test")


if __name__ == "__main__":
    main()