#!/usr/bin/env python3
"""
Minimal Brain Demo Launcher

A simple picker to run any demo from the project root with clean imports.
This eliminates all subfolder import issues and provides a clean demo experience.
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check which demos are available based on dependencies."""
    
    available_demos = []
    
    print("🔍 Checking demo dependencies...")
    
    # Core demos (always available)
    available_demos.extend([
        'text_demo',
        'spatial_demo', 
        'brain_tests'
    ])
    
    # GUI-based demos
    try:
        import pygame
        available_demos.append('grid_debug_demo')
        available_demos.append('wireframe_demo')
        print("✅ pygame available - 2D/3D demos ready")
    except ImportError:
        print("⚠️  pygame not available - install with: pip install pygame")
    
    try:
        import matplotlib
        available_demos.append('scientific_demo')
        print("✅ matplotlib available - Scientific 3D demo ready")
    except ImportError:
        print("⚠️  matplotlib not available - install with: pip install matplotlib")
    
    return available_demos

def run_demo(demo_name, command, description):
    """Run a demo with proper error handling."""
    
    print(f"\n🚀 Starting {demo_name}")
    print(f"   {description}")
    print(f"   Command: {command}")
    print("-" * 50)
    
    try:
        # Run the demo
        result = subprocess.run(command, shell=True, check=True)
        print(f"\n✅ {demo_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {demo_name} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {demo_name} interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ {demo_name} failed: {e}")
        return False

def show_demo_menu():
    """Show the interactive demo selection menu."""
    
    demos = {
        '1': {
            'name': 'Core Brain Tests',
            'command': 'python3 tests/test_minimal_brain.py',
            'description': 'Validate all 4 brain systems with comprehensive tests',
            'always_available': True
        },
        '2': {
            'name': 'Text ASCII Demo',
            'command': 'python3 -c "from demos.picar_x.picar_x_text_demo import main; main()"',
            'description': 'ASCII robot simulation - works everywhere',
            'always_available': True
        },
        '3': {
            'name': 'Spatial Learning Demo', 
            'command': 'python3 -c "from demos.spatial_learning_demo import main; main()"',
            'description': 'Watch spatial navigation emerge from similarity matching',
            'always_available': True
        },
        '4': {
            'name': '2D Grid Debug Demo',
            'command': 'python3 -c "from demos.picar_x.picar_x_2d_debug_demo import main; main()"',
            'description': 'Large grid view for debugging navigation errors',
            'requires': 'pygame'
        },
        '5': {
            'name': 'Wireframe 3D Demo',
            'command': 'python3 -c "from demos.picar_x.picar_x_wireframe_demo import main; main()"',
            'description': 'Battlezone-style first-person robot perspective',
            'requires': 'pygame'
        },
        '6': {
            'name': 'Scientific 3D Demo',
            'command': 'python3 -c "from demos.picar_x.picar_x_3d_demo import main; main()"',
            'description': 'Detailed matplotlib 3D analysis with brain metrics',
            'requires': 'matplotlib'
        },
        '7': {
            'name': 'Brain Server',
            'command': 'python3 server/brain_server.py',
            'description': 'Start TCP server for client-server robot deployment',
            'always_available': True
        }
    }
    
    # Check dependencies
    available_demos = check_dependencies()
    has_pygame = any('pygame' in str(x) for x in available_demos if 'demo' in str(x))
    has_matplotlib = 'scientific_demo' in available_demos
    
    while True:
        print("\n" + "="*60)
        print("🎮 MINIMAL BRAIN DEMO LAUNCHER")
        print("="*60)
        print("Select a demo to run:")
        print()
        
        for key, demo in demos.items():
            # Check if demo is available
            if demo.get('always_available', False):
                status = "✅"
            elif demo.get('requires') == 'pygame' and has_pygame:
                status = "✅"
            elif demo.get('requires') == 'matplotlib' and has_matplotlib:
                status = "✅"
            elif demo.get('requires'):
                status = "❌"
            else:
                status = "✅"
            
            print(f"   {key}. {status} {demo['name']}")
            print(f"      {demo['description']}")
            if demo.get('requires') and status == "❌":
                print(f"      (Requires: {demo['requires']})")
            print()
        
        print("   q. Quit")
        print()
        
        choice = input("Enter choice (1-7, q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Thanks for using the minimal brain!")
            break
        
        if choice in demos:
            demo = demos[choice]
            
            # Check if demo is available
            available = True
            if demo.get('requires') == 'pygame' and not has_pygame:
                available = False
                print(f"\n❌ {demo['name']} requires pygame")
                print("   Install with: pip install pygame")
                continue
            elif demo.get('requires') == 'matplotlib' and not has_matplotlib:
                available = False
                print(f"\n❌ {demo['name']} requires matplotlib")
                print("   Install with: pip install matplotlib")
                continue
            
            if available:
                success = run_demo(demo['name'], demo['command'], demo['description'])
                
                if success:
                    print("\n🎉 Demo completed! Returning to menu...")
                else:
                    print("\n⚠️  Demo ended. Returning to menu...")
                
                input("\nPress Enter to continue...")
        else:
            print("❌ Invalid choice. Please try again.")

def quick_test():
    """Run a quick test of core functionality."""
    
    print("🧪 QUICK BRAIN TEST")
    print("="*25)
    
    try:
        from server.src.brain_factory import MinimalBrain
        
        # Test brain creation
        brain = MinimalBrain()
        print("✅ Brain creation successful")
        
        # Test brain processing
        action, state = brain.process_sensory_input([1, 2, 3, 4])
        print(f"✅ Brain processing successful")
        print(f"   Action: {action[:2]}...")
        print(f"   Method: {state['prediction_method']}")
        
        # Test experience storage
        brain.store_experience([1, 2, 3, 4], action, [1.5, 2.5, 3.5, 4.5])
        print(f"✅ Experience storage successful")
        print(f"   Total experiences: {brain.total_experiences}")
        
        print("\n🎉 All core systems working perfectly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo launcher."""
    
    print("🚀 MINIMAL BRAIN PROJECT")
    print("Emergent Intelligence from 4 Simple Systems")
    print()
    
    # Quick validation
    if not os.path.exists("server/src/brain.py"):
        print("❌ Error: Run this from the brain/ project root directory")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Offer quick test or full menu
    print("Options:")
    print("   1. Quick brain test (validate core functionality)")
    print("   2. Interactive demo menu (choose specific demos)")
    print("   3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        print()
        success = quick_test()
        if success:
            print("\n🎯 Core brain is working! Ready for full demos.")
            cont = input("Continue to demo menu? (y/n): ").strip().lower()
            if cont == 'y':
                show_demo_menu()
    elif choice == '2':
        show_demo_menu()
    elif choice == '3':
        print("\n👋 Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo launcher interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo launcher failed: {e}")
        import traceback
        traceback.print_exc()