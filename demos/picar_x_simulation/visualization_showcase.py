#!/usr/bin/env python3
"""
PiCar-X Visualization Showcase

Demonstrates the different visualization approaches available for the minimal brain robot:
1. Battlezone-style 3D first-person view (like the 1980 game)
2. 2D grid debug view for navigation error analysis
3. Original 3D matplotlib scientific visualization
4. Text-based ASCII visualization

Each serves different purposes for understanding robot behavior and brain learning.
"""

import sys
import os

# Add the brain/ directory to import minimal as a package
current_dir = os.path.dirname(__file__)  # picar_x/
demos_dir = os.path.dirname(current_dir)  # demos/
minimal_dir = os.path.dirname(demos_dir)  # minimal/
brain_dir = os.path.dirname(minimal_dir)   # brain/
sys.path.insert(0, brain_dir)

def show_intro():
    """Show introduction to the visualization showcase."""
    
    print("🎮 PiCar-X VISUALIZATION SHOWCASE")
    print("="*50)
    print()
    print("The minimal brain robot can be visualized in multiple ways,")
    print("each serving different purposes for understanding AI behavior:")
    print()
    
    demos = [
        {
            'name': 'Wireframe 3D (Battlezone Style)',
            'file': 'picar_x_wireframe_demo.py',
            'description': 'Classic 1980 wireframe graphics showing robot perspective',
            'purpose': 'Understand what the robot "sees" and experiences',
            'tech': 'pygame + 3D projection math',
            'inspiration': 'Battlezone (1980) vector graphics',
            'best_for': 'Intuitive understanding of robot perception'
        },
        {
            'name': '2D Grid Debug View',  
            'file': 'picar_x_2d_debug_demo.py',
            'description': 'Top-down grid view with color-coded learning states',
            'purpose': 'Debug navigation errors and spot learning issues',
            'tech': 'pygame 2D graphics',
            'inspiration': 'Classic grid-based game debug views',
            'best_for': 'Finding navigation bugs and optimization'
        },
        {
            'name': 'Scientific 3D Matplotlib',
            'file': 'picar_x_3d_demo.py', 
            'description': 'Detailed 3D plots with brain activity graphs',
            'purpose': 'Scientific analysis of learning performance',
            'tech': 'matplotlib + mpl_toolkits.mplot3d',
            'inspiration': 'Scientific visualization standards',
            'best_for': 'Research analysis and publication'
        },
        {
            'name': 'Text ASCII Visualization',
            'file': 'picar_x_text_demo.py',
            'description': 'ASCII art robot world with real-time stats',  
            'purpose': 'Quick testing without graphics dependencies',
            'tech': 'Pure text output',
            'inspiration': 'Classic terminal-based games',
            'best_for': 'Server environments and quick debugging'
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"{i}. {demo['name']}")
        print(f"   📄 File: {demo['file']}")
        print(f"   📝 Purpose: {demo['purpose']}")
        print(f"   🔧 Tech: {demo['tech']}")
        print(f"   💡 Best for: {demo['best_for']}")
        print()
    
    print("Each visualization shows the same minimal brain learning to navigate,")
    print("but highlights different aspects of the robot's behavior and intelligence.")
    print()

def check_dependencies():
    """Check which visualization demos are available."""
    
    print("🔍 CHECKING DEPENDENCIES")
    print("="*30)
    
    available_demos = []
    
    # Check pygame
    try:
        import pygame
        print("✅ pygame available - 2D Grid Debug View ready")
        available_demos.append('2D Debug')
    except ImportError:
        print("❌ pygame not available - install with: pip install pygame")
    
    # Check pyglet  
    try:
        import pyglet
        import pyglet.gl
        print("✅ pyglet + OpenGL available - Battlezone 3D View ready")
        available_demos.append('Battlezone 3D')
    except ImportError:
        print("❌ pyglet not available - install with: pip install pyglet")
    
    # Check matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        print("✅ matplotlib available - Scientific 3D View ready")
        available_demos.append('Scientific 3D')
    except ImportError:
        print("❌ matplotlib not available - install with: pip install matplotlib")
    
    # Text demo always available
    print("✅ Text ASCII View always available (no dependencies)")
    available_demos.append('Text ASCII')
    
    print(f"\n📊 {len(available_demos)}/4 visualization demos available")
    return available_demos

def run_interactive_menu():
    """Run interactive menu to select and launch demos."""
    
    print("\n🎯 INTERACTIVE DEMO LAUNCHER")
    print("="*35)
    
    options = [
        ("1", "Wireframe 3D Demo (60s)", "python3 picar_x_wireframe_demo.py"),
        ("2", "2D Grid Debug Demo (60s)", "python3 picar_x_2d_debug_demo.py"), 
        ("3", "Scientific 3D Demo (30s)", "python3 picar_x_3d_demo.py"),
        ("4", "Text ASCII Demo (30s)", "python3 picar_x_text_demo.py"),
        ("5", "Quick Test All Demos", "run_quick_tests()"),
        ("q", "Quit", None)
    ]
    
    while True:
        print("\nSelect a demo to run:")
        for key, desc, _ in options:
            print(f"   {key}. {desc}")
        
        choice = input("\nEnter choice (1-5, q): ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        
        # Find matching option
        selected = None
        for key, desc, cmd in options:
            if choice == key:
                selected = (key, desc, cmd)
                break
        
        if not selected:
            print("❌ Invalid choice. Please try again.")
            continue
        
        key, desc, cmd = selected
        
        if cmd is None:  # Quit
            break
        elif cmd == "run_quick_tests()":
            run_quick_tests()
        else:
            print(f"\n🚀 Launching: {desc}")
            print(f"   Command: {cmd}")
            print("   (Note: In CLI mode, you'll need to run this manually)")
            print(f"   Full path: cd minimal/demos/picar_x && {cmd}")
            
            # Ask if user wants to continue or quit
            continue_choice = input("\nContinue with menu? (y/n): ").strip().lower()
            if continue_choice == 'n':
                break

def run_quick_tests():
    """Run quick validation tests for all demos."""
    
    print("\n🧪 RUNNING QUICK VALIDATION TESTS")
    print("="*40)
    
    tests = [
        ("Text Demo Import", "import picar_x_text_demo; print('✅ Text demo ready')"),
        ("2D Debug Import", "import picar_x_2d_debug_demo; print('✅ 2D debug ready')"),
        ("Scientific 3D Import", "import picar_x_3d_demo; print('✅ Scientific 3D ready')"),
        ("Wireframe 3D Import", "import picar_x_wireframe_demo; print('✅ Wireframe 3D ready')"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_code in tests:
        try:
            print(f"\n🔍 {test_name}...")
            exec(test_code)
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} demos validated")
    
    if passed == total:
        print("🎉 All visualization demos are ready to run!")
        print("\nYou can launch any demo by running:")
        print("   cd minimal/demos/picar_x")
        print("   python3 <demo_file>.py")
    else:
        print("⚠️  Some demos may need dependency installation")

def show_comparison():
    """Show detailed comparison of visualization approaches."""
    
    print("\n📊 VISUALIZATION COMPARISON")
    print("="*35)
    print()
    
    comparison = [
        ("Aspect", "Battlezone 3D", "2D Grid Debug", "Scientific 3D", "Text ASCII"),
        ("Purpose", "Robot perspective", "Error debugging", "Research analysis", "Quick testing"),
        ("View", "First-person", "Top-down", "Multiple angles", "ASCII art"),
        ("Performance", "Real-time", "Real-time", "Good", "Excellent"),
        ("Dependencies", "pyglet", "pygame", "matplotlib", "None"),
        ("Best for", "Understanding", "Debugging", "Analysis", "Servers"),
        ("Aesthetics", "Retro/Cool", "Clear/Functional", "Professional", "Minimal"),
        ("Learning curve", "Intuitive", "Easy", "Moderate", "Immediate"),
    ]
    
    # Print comparison table
    for row in comparison:
        print(f"{row[0]:12} | {row[1]:15} | {row[2]:13} | {row[3]:13} | {row[4]:12}")
        if row[0] == "Aspect":
            print("-" * 85)
    
    print("\n💡 Recommendation:")
    print("   • Start with Text ASCII for quick verification")
    print("   • Use 2D Grid Debug for finding navigation issues") 
    print("   • Try Battlezone 3D for the coolest robot perspective")
    print("   • Use Scientific 3D for detailed analysis")

def main():
    """Main showcase function."""
    
    show_intro()
    available = check_dependencies()
    show_comparison()
    
    print("\n" + "="*50)
    print("🎮 The minimal brain robot visualization showcase is ready!")
    print("Each demo shows the same 4-system brain learning to navigate,")
    print("but with different visual perspectives and analysis tools.")
    print("="*50)
    
    # Ask if user wants interactive menu
    choice = input("\nRun interactive demo launcher? (y/n): ").strip().lower()
    if choice == 'y':
        run_interactive_menu()
    else:
        print("\n📋 Manual Launch Commands:")
        print("   cd minimal/demos/picar_x")
        print("   python3 picar_x_wireframe_demo.py       # Wireframe 3D")
        print("   python3 picar_x_2d_debug_demo.py        # 2D Grid Debug")  
        print("   python3 picar_x_3d_demo.py              # Scientific 3D")
        print("   python3 picar_x_text_demo.py            # Text ASCII")

if __name__ == "__main__":
    main()