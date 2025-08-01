#!/usr/bin/env python3
"""
Field-Native Intelligence Demo

An interactive demonstration of the brain's emergent intelligence through
continuous field dynamics and predictive processing.

This demo showcases:
- Self-modifying field dynamics
- Predictive processing in action
- Emergent behaviors from simple rules
- Real-time learning and adaptation

Usage:
    python3 demo.py                    # Run interactive visualization
    python3 demo.py --mode terminal    # Terminal-only mode
    python3 demo.py --mode server      # Server mode (for external clients)
"""

import sys
import os
import time
import subprocess
import signal
import argparse
from pathlib import Path

# Add server to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

def check_dependencies():
    """Check and report on available dependencies."""
    deps = {
        'pygame': {'available': False, 'purpose': '2D visualization'},
        'matplotlib': {'available': False, 'purpose': '3D plots'},
        'numpy': {'available': False, 'purpose': 'core computation'},
        'torch': {'available': False, 'purpose': 'tensor operations'}
    }
    
    # Check each dependency
    for module, info in deps.items():
        try:
            __import__(module)
            info['available'] = True
        except ImportError:
            pass
    
    # Report status
    all_available = all(d['available'] for d in deps.values())
    required = ['numpy', 'torch']
    required_available = all(deps[m]['available'] for m in required)
    
    if not required_available:
        print("❌ Missing required dependencies:")
        for module in required:
            if not deps[module]['available']:
                print(f"   - {module}: {deps[module]['purpose']}")
        print("\nInstall with: pip install numpy torch")
        sys.exit(1)
    
    return all_available, deps

def run_terminal_demo():
    """Run a terminal-based demonstration."""
    print("\n" + "="*60)
    print("TERMINAL DEMONSTRATION")
    print("="*60)
    
    from src.brains.field.unified_field_brain import UnifiedFieldBrain
    import numpy as np
    
    print("\n1. Creating brain...")
    brain = UnifiedFieldBrain(
        sensory_dim=8,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=False
    )
    
    print("\n2. Testing basic responsiveness...")
    for i in range(5):
        sensory = [0.5 + 0.2 * np.sin(i * 0.5 + j * 0.1) for j in range(8)]
        motor, state = brain.process_robot_cycle(sensory)
        print(f"   Cycle {i+1}: confidence={state.get('confidence', 0):.2f}, "
              f"motor=[{', '.join(f'{m:.2f}' for m in motor)}]")
    
    print("\n3. Testing pattern learning...")
    pattern = [[1.0 if j == i % 8 else 0.0 for j in range(8)] for i in range(4)]
    
    for epoch in range(3):
        print(f"\n   Epoch {epoch + 1}:")
        for i, p in enumerate(pattern):
            motor, state = brain.process_robot_cycle(p)
            print(f"     Pattern {i+1}: energy={state.get('energy', 0):.2f}")
    
    print("\n4. Field statistics:")
    props = brain.field_dynamics.get_emergent_properties()
    print(f"   Evolution cycles: {brain.field_dynamics.evolution_count}")
    print(f"   Self-modification: {brain.field_dynamics.self_modification_strength:.3f}")
    print(f"   Confidence: {props['smoothed_confidence']:.3f}")
    print(f"   Active regions: {len(brain.topology_region_system.regions)}")
    
    print("\n✅ Terminal demo complete!")

def run_visual_demo():
    """Run the visual demonstration with pygame."""
    try:
        import pygame
        import numpy as np
        from src.brains.field.unified_field_brain import UnifiedFieldBrain
        
        # Initialize pygame
        pygame.init()
        width, height = 1200, 800
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Field-Native Intelligence Demo")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
        
        # Create brain
        brain = UnifiedFieldBrain(
            sensory_dim=8,
            motor_dim=2,  # 2D movement
            spatial_resolution=32,
            quiet_mode=True
        )
        
        # Virtual robot state
        robot_x, robot_y = width // 2, height // 2
        robot_angle = 0
        trail = []
        
        # Light sources (goals)
        lights = [
            {'x': width * 0.25, 'y': height * 0.25, 'intensity': 1.0},
            {'x': width * 0.75, 'y': height * 0.75, 'intensity': 1.0}
        ]
        
        # Main loop
        running = True
        cycle = 0
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Reset robot position
                        robot_x, robot_y = width // 2, height // 2
                        trail.clear()
                    elif event.key == pygame.K_r:
                        # Add random light
                        lights.append({
                            'x': np.random.randint(100, width-100),
                            'y': np.random.randint(100, height-100),
                            'intensity': np.random.uniform(0.5, 1.0)
                        })
            
            # Calculate sensory input based on lights
            sensory = []
            for i in range(8):
                angle = robot_angle + (i * 45)  # 8 sensors, 45 degrees apart
                sensor_x = robot_x + np.cos(np.radians(angle)) * 50
                sensor_y = robot_y + np.sin(np.radians(angle)) * 50
                
                # Calculate light intensity at sensor
                intensity = 0
                for light in lights:
                    dist = np.sqrt((sensor_x - light['x'])**2 + (sensor_y - light['y'])**2)
                    intensity += light['intensity'] * max(0, 1 - dist / 300)
                
                sensory.append(min(1.0, intensity))
            
            # Process through brain
            motor, brain_state = brain.process_robot_cycle(sensory)
            
            # Update robot position
            if len(motor) >= 2:
                # Motor[0] = forward/backward, Motor[1] = turn
                speed = motor[0] * 5
                robot_x += np.cos(np.radians(robot_angle)) * speed
                robot_y += np.sin(np.radians(robot_angle)) * speed
                robot_angle += motor[1] * 10
                
                # Keep robot on screen
                robot_x = max(50, min(width - 50, robot_x))
                robot_y = max(50, min(height - 50, robot_y))
                
                # Add to trail
                trail.append((robot_x, robot_y))
                if len(trail) > 200:
                    trail.pop(0)
            
            # Clear screen
            screen.fill((20, 20, 30))
            
            # Draw lights
            for light in lights:
                intensity = int(light['intensity'] * 255)
                pygame.draw.circle(screen, (intensity, intensity, 100), 
                                 (int(light['x']), int(light['y'])), 30)
                # Light glow
                for r in range(30, 100, 10):
                    alpha = int((100 - r) * 2)
                    pygame.draw.circle(screen, (alpha, alpha, 50), 
                                     (int(light['x']), int(light['y'])), r, 1)
            
            # Draw trail
            for i in range(1, len(trail)):
                alpha = int(255 * (i / len(trail)))
                color = (alpha // 4, alpha // 2, alpha)
                pygame.draw.line(screen, color, trail[i-1], trail[i], 2)
            
            # Draw robot
            robot_color = (100, 200, 100)  # Default green
            confidence = brain_state.get('confidence', 0)
            if confidence > 0.7:
                robot_color = (100, 100, 200)  # Blue when confident
            elif confidence < 0.3:
                robot_color = (200, 100, 100)  # Red when uncertain
            
            pygame.draw.circle(screen, robot_color, (int(robot_x), int(robot_y)), 20)
            
            # Draw robot direction
            end_x = robot_x + np.cos(np.radians(robot_angle)) * 30
            end_y = robot_y + np.sin(np.radians(robot_angle)) * 30
            pygame.draw.line(screen, (255, 255, 255), 
                           (robot_x, robot_y), (end_x, end_y), 3)
            
            # Draw sensor rays
            for i in range(8):
                angle = robot_angle + (i * 45)
                sensor_x = robot_x + np.cos(np.radians(angle)) * 50
                sensor_y = robot_y + np.sin(np.radians(angle)) * 50
                intensity = int(sensory[i] * 255)
                pygame.draw.line(screen, (intensity, intensity, 0), 
                               (robot_x, robot_y), (sensor_x, sensor_y), 1)
            
            # Draw HUD
            y_offset = 10
            texts = [
                f"Field-Native Intelligence Demo",
                f"Cycle: {cycle}",
                f"Confidence: {confidence:.2f}",
                f"Energy: {brain_state.get('energy', 0):.2f}",
                f"Information: {brain_state.get('information', 0):.2f}",
                f"Active Regions: {len(brain.topology_region_system.regions)}",
                "",
                "Controls:",
                "SPACE - Reset position",
                "R - Add random light",
                "ESC - Exit"
            ]
            
            for text in texts:
                if text:  # Skip empty lines
                    surface = font.render(text, True, (255, 255, 255))
                    screen.blit(surface, (10, y_offset))
                y_offset += 25
            
            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
            cycle += 1
        
        pygame.quit()
        print("\n✅ Visual demo complete!")
        
    except ImportError as e:
        print(f"\n❌ Cannot run visual demo: {e}")
        print("Install pygame with: pip install pygame")
        print("\nFalling back to terminal demo...")
        run_terminal_demo()

def run_server_demo():
    """Run the brain server for external connections."""
    print("\n" + "="*60)
    print("BRAIN SERVER MODE")
    print("="*60)
    
    # Kill any existing servers
    subprocess.run(['pkill', '-f', 'brain.py'], stderr=subprocess.DEVNULL)
    time.sleep(1)
    
    # Start the brain server
    print("\nStarting brain server...")
    server_process = subprocess.Popen(
        [sys.executable, 'server/brain.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("✅ Brain server started on port 9999")
    print("\nYou can now connect with:")
    print("  - Python clients using MinimalBrainClient")
    print("  - The biological embodied learning experiment")
    print("  - Custom robot simulations")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Wait for interrupt
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        server_process.terminate()
        time.sleep(1)
        if server_process.poll() is None:
            server_process.kill()
        print("✅ Server stopped")

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="Field-Native Intelligence Demo")
    parser.add_argument('--mode', choices=['visual', 'terminal', 'server'], 
                       default='visual', help='Demo mode to run')
    args = parser.parse_args()
    
    print("\n🧠 FIELD-NATIVE INTELLIGENCE SYSTEM")
    print("="*60)
    print("A continuous field-based artificial brain that combines")
    print("predictive processing with self-modifying dynamics.")
    print("="*60)
    
    # Check dependencies
    all_deps, deps = check_dependencies()
    
    if args.mode == 'visual' and not deps['pygame']['available']:
        print("\n⚠️  Pygame not available, falling back to terminal mode")
        args.mode = 'terminal'
    
    # Run appropriate demo
    if args.mode == 'visual':
        print("\n🎮 Starting visual demo...")
        run_visual_demo()
    elif args.mode == 'terminal':
        print("\n📟 Starting terminal demo...")
        run_terminal_demo()
    elif args.mode == 'server':
        print("\n🌐 Starting server mode...")
        run_server_demo()

if __name__ == "__main__":
    main()