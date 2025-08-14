#!/usr/bin/env python3
"""
Test brainstem in reflex-only mode (no brain connection).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_reflexes():
    """Test brainstem reflexes without brain."""
    print("=" * 60)
    print("BRAINSTEM REFLEX TEST")
    print("=" * 60)
    print("\nTesting brainstem without brain connection...")
    print("This tests safety reflexes and basic operation.\n")
    
    from src.brainstem.brainstem import Brainstem
    
    try:
        # Create brainstem without brain connection
        print("Creating brainstem (no brain)...")
        brainstem = Brainstem(
            brain_host="localhost",  # Won't connect
            brain_port=9999,
            enable_brain=False  # Disable brain connection
        )
        
        print("\n" + "=" * 60)
        print("TEST: Run 10 cycles")
        print("=" * 60)
        
        for i in range(10):
            print(f"\nCycle {i+1}:")
            
            # Run one cycle
            try:
                brainstem.cycle()
                print("  ✅ Cycle completed")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(0.1)
        
        print("\n" + "=" * 60)
        print("TEST: Check motor command creation")
        print("=" * 60)
        
        # Test motor command creation directly
        brain_output = [0.2, -0.2, 0.0, 0.0, 0.0, 0.0]
        motor_cmd, servo_cmd = brainstem.brain_to_hardware_commands(brain_output)
        
        print(f"Brain output: {brain_output[:2]}")
        print(f"Motor command:")
        print(f"  left_pwm_duty: {motor_cmd.left_pwm_duty}")
        print(f"  right_pwm_duty: {motor_cmd.right_pwm_duty}")
        print(f"Servo command:")
        print(f"  steering_pw: {servo_cmd.steering_pw}")
        print(f"  camera_pan_pw: {servo_cmd.camera_pan_pw}")
        print(f"  camera_tilt_pw: {servo_cmd.camera_tilt_pw}")
        
        # Try to access monitor attributes
        if brainstem.monitor:
            print("\n" + "=" * 60)
            print("TEST: Monitor update")
            print("=" * 60)
            
            try:
                brainstem.monitor.update_motors(
                    left=motor_cmd.left_pwm_duty,
                    right=motor_cmd.right_pwm_duty,
                    steering=servo_cmd.steering_pw,
                    camera=servo_cmd.camera_pan_pw
                )
                print("✅ Monitor update successful")
            except AttributeError as e:
                print(f"❌ AttributeError: {e}")
                print(f"  motor_cmd attributes: {dir(motor_cmd)}")
        
        brainstem.shutdown()
        print("\n✅ Test complete")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reflexes()