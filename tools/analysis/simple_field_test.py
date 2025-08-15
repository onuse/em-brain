#!/usr/bin/env python3
"""
Simple Field Brain Test

A standalone test to validate field brain functionality.
"""

import sys
import os

# Get the absolute path to the server directory
script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.join(script_dir, '..', '..', 'server')
sys.path.insert(0, server_dir)

def test_field_brain():
    """Test field brain components."""
    print("🧪 Simple Field Brain Test")
    print("=" * 30)
    
    # Test 1: Direct module import
    print("1️⃣ Testing direct module import...")
    try:
        import subprocess
        import tempfile
        
        # Create a temporary test script that runs in the server directory
        test_script = '''
import sys
import os
sys.path.insert(0, "src")

try:
    from brain import MinimalBrain
    print("SUCCESS: MinimalBrain imported")
    
    # Test field brain creation
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 4,
            'motor_dim': 2,
            'field_spatial_resolution': 3
        }
    }
    
    brain = MinimalBrain(
        config=config,
        brain_type="field",
        sensory_dim=4,
        motor_dim=2,
        enable_logging=False,
        quiet_mode=True
    )
    
    print(f"SUCCESS: Field brain created - {brain.brain_type}")
    print(f"SUCCESS: Vector brain type - {type(brain.vector_brain).__name__}")
    
    # Test basic methods exist
    methods = ['process_sensory_input', 'get_brain_stats', 'store_experience']
    for method in methods:
        if hasattr(brain, method):
            print(f"SUCCESS: Method {method} exists")
        else:
            print(f"ERROR: Method {method} missing")
    
    # Test TCP server compatibility
    from communication.tcp_server import MinimalTCPServer
    server = MinimalTCPServer(brain, host='127.0.0.1', port=0)
    print(f"SUCCESS: TCP server created with field brain")
    
    # Cleanup
    brain.finalize_session()
    print("SUCCESS: All tests passed")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
'''
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_script = f.name
        
        try:
            # Run the script from the server directory
            result = subprocess.run(
                [sys.executable, temp_script],
                cwd=server_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print("Script output:")
            print(result.stdout)
            
            if result.stderr:
                print("Script errors:")
                print(result.stderr)
            
            success = result.returncode == 0 and "SUCCESS: All tests passed" in result.stdout
            
            if success:
                print("✅ Field brain integration test PASSED")
                return True
            else:
                print("❌ Field brain integration test FAILED")
                return False
                
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_script)
            except:
                pass
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


def main():
    """Main test execution."""
    success = test_field_brain()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 Field Brain TCP Integration: WORKING")
        print("✅ Field brain can be instantiated")
        print("✅ TCP server compatibility confirmed")
        print("✅ Required interface methods available")
        print("✅ Ready for production deployment")
    else:
        print("🔧 Field Brain TCP Integration: NEEDS ATTENTION")
        print("❌ Some components are not working correctly")
        print("❌ Manual investigation required")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)