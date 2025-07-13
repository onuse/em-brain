#!/usr/bin/env python3
"""
Quick Demo Runner

Run specific demos directly from command line without interactive menu.
Usage: python3 demo_runner.py <demo_name>
"""

import sys
import subprocess

def main():
    """Run specific demo by name."""
    
    demos = {
        'brain': 'python3 -c "from tests.test_minimal_brain import main; main()"',
        'test_demo': 'python3 -c "from demos.test_demo import main; main()"',
        'demo_2d': 'python3 -c "from demos.demo_2d import main; main()"',
        'demo_3d': 'python3 -c "from demos.demo_3d import main; main()"',
        'demo_3d_hifi': 'python3 -c "from demos.demo_3d_new import main; main()"',
        'spatial_learning': 'python3 -c "from demos.spatial_learning_demo import main; main()"',
        'server': 'python3 brain_server.py',
        # Old names for backward compatibility
        'text': 'python3 -c "from demos.picar_x_simulation.picar_x_text_demo import main; main()"',
        'grid': 'python3 -c "from demos.picar_x_simulation.picar_x_2d_debug_demo import main; main()"',
        'wireframe': 'python3 -c "from demos.picar_x_simulation.picar_x_wireframe_demo import main; main()"',
        'scientific': 'python3 -c "from demos.picar_x_simulation.picar_x_3d_demo import main; main()"'
    }
    
    if len(sys.argv) != 2:
        print("🎮 Quick Demo Runner")
        print("\nUsage: python3 demo_runner.py <demo_name>")
        print("\nAvailable demos:")
        for name, cmd in demos.items():
            print(f"   {name:12} - {cmd}")
        print("\nExample: python3 demo_runner.py text")
        return
    
    demo_name = sys.argv[1].lower()
    
    if demo_name not in demos:
        print(f"❌ Unknown demo: {demo_name}")
        print(f"Available: {', '.join(demos.keys())}")
        return
    
    command = demos[demo_name]
    print(f"🚀 Running {demo_name} demo...")
    print(f"   Command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {demo_name} demo completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {demo_name} demo failed with return code {e.returncode}")
    except KeyboardInterrupt:
        print(f"⏹️  {demo_name} demo interrupted")

if __name__ == "__main__":
    main()