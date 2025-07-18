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
        'brain': 'python3 server/tests/test_minimal_brain.py',
        'test_demo': 'python3 -c "from demos.test_demo import main; main()"',
        'demo_2d': 'python3 -c "from demos.demo_2d import main; main()"',
        'demo_3d': 'python3 -c "from demos.demo_3d import main; main()"',
        'demo_3d_hifi': 'python3 -c "from demos.demo_3d_new import main; main()"',
        'spatial_learning': 'python3 -c "from demos.spatial_learning_demo import main; main()"',
        'server': 'python3 server/brain_server.py',
        'vocal_test': 'python3 client_picarx/test_vocal_system.py',
        'vocal_mac': 'python3 client_picarx/test_vocal_system.py mac',
        'vocal_visual': 'python3 client_picarx/vocal_demo_visual.py',
        'vocal_diagnostic': 'python3 client_picarx/vocal_demo_diagnostic.py',
        'hardware_scan': 'python3 client_picarx/docs/installation/hardware_discovery.py',
        'hardware_report': 'python3 client_picarx/docs/installation/hardware_discovery.py --save hardware_report.txt',
        'motivation_test': 'python3 server/test_motivation_system.py',
        'motivation_balance': 'python3 server/test_motivation_balance.py',
        'motivation_integration': 'python3 server/test_motivation_integration.py',
        'embodied_free_energy': 'python3 server/test_embodied_free_energy.py',
        '3d_embodied': 'python3 -c "from demos.demo_3d_embodied import main; main()"',
        'multimodal': 'python3 -c "from demos.demo_3d_multimodal import main; main()"',
        'socket_embodied': 'python3 demos/picar_x_simulation/pure_socket_embodied_brainstem.py',
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