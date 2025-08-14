#!/usr/bin/env python3
"""
I2C Hardware Diagnostic Tool for PiCar-X

This script helps diagnose why the PCA9685 servo controller is not detected.
The PCA9685 should appear at address 0x40 but is missing.

Expected I2C devices:
- 0x14: ADC (Analog-to-Digital Converter) - DETECTED ✓
- 0x40: PCA9685 (16-channel PWM/Servo controller) - MISSING ✗

Without the PCA9685, motors and servos cannot be controlled.
"""

import subprocess
import sys
import time

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def check_i2c_tools():
    """Check if i2c-tools is installed."""
    print("🔍 Checking for i2c-tools...")
    stdout, stderr, code = run_command("which i2cdetect")
    if code != 0:
        print("   ❌ i2c-tools not installed")
        print("   Install with: sudo apt-get install i2c-tools")
        return False
    print("   ✓ i2c-tools found")
    return True

def check_i2c_enabled():
    """Check if I2C is enabled in config."""
    print("\n🔍 Checking I2C configuration...")
    
    # Check if I2C is enabled in boot config
    stdout, stderr, code = run_command("grep -E '^dtparam=i2c_arm=on' /boot/config.txt")
    if code != 0:
        print("   ⚠️ I2C may not be enabled in /boot/config.txt")
        print("   Enable with: sudo raspi-config → Interface Options → I2C")
    else:
        print("   ✓ I2C enabled in boot config")
    
    # Check if I2C modules are loaded
    stdout, stderr, code = run_command("lsmod | grep i2c")
    if "i2c_bcm" in stdout or "i2c_dev" in stdout:
        print("   ✓ I2C kernel modules loaded")
    else:
        print("   ⚠️ I2C kernel modules not loaded")
        print("   Try: sudo modprobe i2c-dev")
    
    return True

def scan_i2c_buses():
    """Scan all I2C buses for devices."""
    print("\n🔍 Scanning I2C buses...")
    
    buses = []
    # Check which I2C buses exist
    stdout, stderr, code = run_command("ls /dev/i2c-*")
    if code == 0:
        for line in stdout.strip().split('\n'):
            if '/dev/i2c-' in line:
                bus_num = line.split('-')[-1]
                buses.append(bus_num)
    
    if not buses:
        print("   ❌ No I2C buses found!")
        return []
    
    print(f"   Found buses: {', '.join(buses)}")
    
    devices = []
    for bus in buses:
        print(f"\n   Bus {bus}:")
        stdout, stderr, code = run_command(f"sudo i2cdetect -y {bus}")
        if code == 0:
            print(stdout)
            # Parse devices from output
            lines = stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) > 1:
                    for part in parts[1:]:
                        if part != '--' and part != 'UU':
                            try:
                                addr = int(part, 16)
                                devices.append((int(bus), addr))
                                print(f"      Device found at 0x{addr:02x}")
                            except:
                                pass
        else:
            print(f"      Error scanning bus {bus}: {stderr}")
    
    return devices

def check_robot_hat():
    """Check Robot HAT specific issues."""
    print("\n🔍 Checking Robot HAT...")
    
    # Check if Robot HAT might need initialization
    print("\n   Robot HAT Troubleshooting:")
    print("   1. Physical connection:")
    print("      - Is the Robot HAT properly seated on GPIO pins?")
    print("      - Are all 40 pins connected?")
    print("      - Is the HAT powered? (check LED indicators)")
    
    print("\n   2. Power issues:")
    print("      - The PCA9685 needs stable power to function")
    print("      - Is the battery connected and charged (>7V)?")
    print("      - Try powering via USB first for testing")
    
    print("\n   3. Robot HAT initialization:")
    print("      - Some Robot HATs need initialization code")
    print("      - The PCA9685 might need a reset sequence")
    
    # Try to probe the PCA9685 directly
    print("\n   Attempting direct PCA9685 probe at 0x40...")
    stdout, stderr, code = run_command("sudo i2cget -y 1 0x40 0x00")
    if code == 0:
        print(f"      ✓ Read from 0x40: {stdout.strip()}")
    else:
        print(f"      ✗ Cannot read from 0x40: {stderr.strip()}")
        
        # Try alternate addresses (some boards use 0x60)
        for addr in [0x60, 0x41, 0x42]:
            stdout, stderr, code = run_command(f"sudo i2cget -y 1 0x{addr:02x} 0x00 2>/dev/null")
            if code == 0:
                print(f"      🔍 Found device at alternate address 0x{addr:02x}!")
                print(f"         You may need to update I2C_ADDRESSES in bare_metal_hal.py")

def suggest_fixes(devices):
    """Suggest fixes based on what was found."""
    print("\n📋 Diagnosis Summary:")
    
    has_adc = False
    has_pca = False
    
    for bus, addr in devices:
        if addr == 0x14:
            has_adc = True
            print(f"   ✓ ADC found at 0x14 on bus {bus}")
        elif addr == 0x40:
            has_pca = True
            print(f"   ✓ PCA9685 found at 0x40 on bus {bus}")
    
    if not has_adc:
        print("   ✗ ADC (0x14) not found - Robot HAT may not be connected")
    
    if not has_pca:
        print("   ✗ PCA9685 (0x40) not found - Servo/motor control unavailable")
    
    print("\n🔧 Recommended Actions:")
    
    if not devices:
        print("   1. Check Robot HAT physical connection")
        print("   2. Enable I2C with: sudo raspi-config")
        print("   3. Reboot after enabling I2C")
    elif has_adc and not has_pca:
        print("   1. Robot HAT partially working (ADC detected)")
        print("   2. PCA9685 specific issue - likely power related")
        print("   3. Try these fixes:")
        print("      a) Connect battery power (7.4V) to Robot HAT")
        print("      b) Check Robot HAT has power LED lit")
        print("      c) Try this initialization script:")
        print("")
        print("      sudo python3 -c \"")
        print("      import smbus")
        print("      bus = smbus.SMBus(1)")
        print("      # Reset PCA9685")
        print("      bus.write_byte_data(0x40, 0x00, 0x06)  # Software reset")
        print("      import time; time.sleep(0.1)")
        print("      bus.write_byte_data(0x40, 0x00, 0x00)  # Normal mode")
        print("      print('PCA9685 reset attempted')")
        print("      \"")
        print("")
        print("   4. If PCA9685 still not detected:")
        print("      - Check if it's at alternate address (0x60)")
        print("      - Robot HAT may be defective")
        print("      - Try external PCA9685 module for testing")
    elif has_adc and has_pca:
        print("   ✅ All devices detected! Robot should work.")
        print("   Update bare_metal_hal.py if needed.")

def check_sunfounder_libs():
    """Check if Sunfounder libraries might help."""
    print("\n🔍 Checking for Sunfounder libraries...")
    
    try:
        import robot_hat
        print("   ✓ robot_hat library found")
        print("   You could try using Sunfounder's initialization:")
        print("      from robot_hat import PWM")
        print("      PWM().period(4095)  # This might initialize PCA9685")
    except ImportError:
        print("   ℹ️ robot_hat not installed (this is OK)")
        print("   We use bare metal control instead")

def main():
    print("=" * 60)
    print("PiCar-X I2C Hardware Diagnostic Tool")
    print("=" * 60)
    
    # Check prerequisites
    if not check_i2c_tools():
        return
    
    check_i2c_enabled()
    
    # Scan for devices
    devices = scan_i2c_buses()
    
    # Robot HAT specific checks
    check_robot_hat()
    
    # Check for Sunfounder libraries
    check_sunfounder_libs()
    
    # Provide diagnosis
    suggest_fixes(devices)
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("\nNext steps:")
    print("1. Follow the recommended actions above")
    print("2. Run this script again after fixes")
    print("3. When PCA9685 is detected at 0x40, the robot should work")
    print("=" * 60)

if __name__ == "__main__":
    main()