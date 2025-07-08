#!/usr/bin/env python3
"""
Debug robot movement to understand why it's stuck.
"""

from simulation.brainstem_sim import GridWorldBrainstem


def debug_robot_world():
    """Debug the robot's world and movement capabilities."""
    print("🔍 Debugging Robot Movement")
    print("==========================")
    
    brainstem = GridWorldBrainstem(
        world_width=12, world_height=12, seed=42, use_sockets=False
    )
    
    # Print initial world state
    world_state = brainstem.get_world_state()
    robot_pos = world_state['robot_position']
    robot_orientation = world_state['robot_orientation']
    
    print(f"Robot position: {robot_pos}")
    print(f"Robot orientation: {robot_orientation} (0=North, 1=East, 2=South, 3=West)")
    
    # Check surrounding cells
    x, y = robot_pos
    grid = world_state['world_grid']
    
    print(f"\nSurrounding cells around ({x}, {y}):")
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                cell_value = grid[nx][ny]
                cell_type = {0: "EMPTY", 1: "WALL", 2: "FOOD", 3: "DANGER"}.get(cell_value, "UNKNOWN")
                marker = "🤖" if (dx, dy) == (0, 0) else "  "
                print(f"  ({nx:2d},{ny:2d}): {cell_type:7s} {marker}")
            else:
                print(f"  ({nx:2d},{ny:2d}): OUT_OF_BOUNDS")
    
    # Test basic movement commands
    print(f"\n🚶 Testing Movement Commands:")
    initial_stats = brainstem.get_simulation_stats()
    print(f"Initial position: {initial_stats['robot_position']}")
    print(f"Initial orientation: {initial_stats['robot_orientation']}")
    
    # Test forward movement
    print(f"\nTesting forward movement...")
    result = brainstem.execute_motor_commands({"forward_motor": 0.5})
    new_stats = brainstem.get_simulation_stats()
    print(f"Result: {result}")
    print(f"New position: {new_stats['robot_position']}")
    print(f"New orientation: {new_stats['robot_orientation']}")
    
    # Test turning
    print(f"\nTesting turn right...")
    result = brainstem.execute_motor_commands({"turn_motor": 0.5})
    new_stats = brainstem.get_simulation_stats()
    print(f"Result: {result}")
    print(f"New position: {new_stats['robot_position']}")  
    print(f"New orientation: {new_stats['robot_orientation']}")
    
    # Test forward again
    print(f"\nTesting forward movement after turn...")
    result = brainstem.execute_motor_commands({"forward_motor": 0.5})
    new_stats = brainstem.get_simulation_stats()
    print(f"Result: {result}")
    print(f"New position: {new_stats['robot_position']}")
    print(f"New orientation: {new_stats['robot_orientation']}")
    
    # Show final world around robot
    final_pos = new_stats['robot_position']
    x, y = final_pos
    print(f"\nFinal surrounding cells around ({x}, {y}):")
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                cell_value = grid[nx][ny]
                cell_type = {0: "EMPTY", 1: "WALL", 2: "FOOD", 3: "DANGER"}.get(cell_value, "UNKNOWN")
                marker = "🤖" if (dx, dy) == (0, 0) else "  "
                print(f"  ({nx:2d},{ny:2d}): {cell_type:7s} {marker}")


if __name__ == "__main__":
    debug_robot_world()