#!/usr/bin/env python3
"""
Fixed Display Demo - Forces correct window sizing for brain monitoring
"""

import pygame
import time
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.integrated_display import IntegratedDisplay


def main():
    """Launch demo with fixed window sizing."""
    print("🧠 Fixed Display Demo - With Brain Monitor")
    print("=" * 50)
    print("Forcing larger window size to show brain panel...")
    print()
    
    # Create smaller world to ensure window fits on screen
    brainstem = GridWorldBrainstem(world_width=12, world_height=12, seed=42)
    
    # Use smaller cell size to fit everything
    cell_size = 25
    
    # Calculate expected window size
    grid_width = brainstem.simulation.width * cell_size
    brain_panel_width = 400
    total_width = grid_width + brain_panel_width
    
    print(f"Grid size: {grid_width}px")
    print(f"Brain panel: {brain_panel_width}px") 
    print(f"Total window width: {total_width}px")
    print()
    print("If you still don't see the brain panel, try:")
    print("1. Moving the window to check if part is off-screen")
    print("2. Maximizing the window")
    print("3. Checking your display resolution")
    print()
    
    # Initialize with persistent memory
    session_id = brainstem.brain_client.start_memory_session("Fixed Display Demo")
    print(f"📝 Started memory session: {session_id}")
    
    try:
        # Create display with explicit sizing
        display = IntegratedDisplay(brainstem, cell_size=cell_size)
        
        # Print actual window dimensions
        print(f"🖥️  Window dimensions: {display.window_width}x{display.window_height}")
        print(f"🚀 Launching GUI...")
        
        # Run with slower speed for observation
        display.run(auto_step=True, step_delay=0.3)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save state
        brainstem.brain_client.save_current_state()
        brainstem.brain_client.end_memory_session()
        print("💾 Session saved")
    
    print("✅ Demo completed")


if __name__ == "__main__":
    main()