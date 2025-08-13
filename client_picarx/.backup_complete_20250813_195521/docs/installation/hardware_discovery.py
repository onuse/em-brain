#!/usr/bin/env python3
"""
Hardware Interface Discovery Script for PiCar-X

Scans Raspberry Pi Zero 2 WH to discover available hardware interfaces
that could be used alongside or instead of SunFounder APIs.

Run this on your Pi Zero after SunFounder installation to see what
lower-level interfaces are available for granular control.

Usage:
    python3 hardware_discovery.py
    python3 hardware_discovery.py --detailed  # More verbose output
    python3 hardware_discovery.py --json      # JSON output for parsing
"""

import sys
import json
import time
import argparse
from typing import Dict, List, Optional

# Hardware interface imports (with fallbacks)
try:
    import smbus
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

try:
    import spidev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False

try:
    from picarx import Picarx
    SUNFOUNDER_AVAILABLE = True
except ImportError:
    SUNFOUNDER_AVAILABLE = False


class HardwareDiscovery:
    """Discover available hardware interfaces on Raspberry Pi."""
    
    def __init__(self, detailed=False):
        self.detailed = detailed
        self.discovered = {
            'summary': {},
            'i2c_devices': {},
            'gpio_capabilities': {},
            'spi_devices': {},
            'sunfounder_status': {},
            'recommendations': []
        }
    
    def scan_all_interfaces(self):
        """Perform complete hardware scan."""
        
        print("🔍 Hardware Interface Discovery")
        print("=" * 50)
        print(f"Platform: Raspberry Pi (Python {sys.version.split()[0]})")
        print()
        
        # Scan different interface types
        self._scan_i2c_bus()
        self._scan_gpio_capabilities() 
        self._scan_spi_interfaces()
        self._test_sunfounder_integration()
        self._generate_recommendations()
        
        return self.discovered
    
    def _scan_i2c_bus(self):
        """Scan I2C bus for connected devices."""
        
        print("🔌 I2C Bus Scanning")
        print("-" * 20)
        
        if not I2C_AVAILABLE:
            print("❌ smbus not available - install with: pip3 install smbus")
            self.discovered['i2c_devices'] = {'error': 'smbus_not_available'}
            return
        
        try:
            bus = smbus.SMBus(1)  # I2C bus 1 (default on Pi)
            found_devices = []
            
            print("Scanning I2C addresses 0x08-0x77...")
            
            for addr in range(0x08, 0x78):
                try:
                    # Try to read from device
                    bus.read_byte(addr)
                    device_info = self._identify_i2c_device(addr)
                    found_devices.append(device_info)
                    
                    if self.detailed:
                        print(f"✅ 0x{addr:02X}: {device_info['description']}")
                    
                except OSError:
                    # Device not responding
                    continue
            
            self.discovered['i2c_devices'] = {
                'bus_available': True,
                'devices_found': len(found_devices),
                'devices': found_devices
            }
            
            print(f"✅ Found {len(found_devices)} I2C devices")
            if not self.detailed and found_devices:
                addresses = [f"0x{d['address']:02X}" for d in found_devices]
                print(f"   Addresses: {', '.join(addresses)}")
            
        except Exception as e:
            print(f"❌ I2C scan failed: {e}")
            self.discovered['i2c_devices'] = {'error': str(e)}
        
        print()
    
    def _identify_i2c_device(self, addr):
        """Try to identify what type of device is at this I2C address."""
        
        # Common I2C device addresses for robotics
        known_devices = {
            0x14: "SunFounder Robot HAT",
            0x48: "ADS1015/ADS1115 ADC",
            0x68: "MPU6050 IMU / DS1307 RTC",
            0x70: "TCA9548A I2C Multiplexer",
            0x77: "BMP280 Pressure Sensor",
            0x40: "PCA9685 PWM Driver",
            0x29: "VL53L0X Distance Sensor",
            0x1D: "ADXL345 Accelerometer"
        }
        
        device_info = {
            'address': addr,
            'hex_address': f"0x{addr:02X}",
            'description': known_devices.get(addr, "Unknown I2C Device"),
            'potential_uses': []
        }
        
        # Add potential uses based on device type
        if addr == 0x14:  # Robot HAT
            device_info['potential_uses'] = [
                'Motor control via I2C',
                'Servo control',
                'ADC readings',
                'GPIO expansion'
            ]
        elif addr in [0x40]:  # PWM driver
            device_info['potential_uses'] = [
                'Individual servo control',
                'LED control',
                'Motor speed control'
            ]
        elif addr in [0x48]:  # ADC
            device_info['potential_uses'] = [
                'Battery voltage monitoring',
                'Current sensing',
                'Analog sensor reading'
            ]
        elif addr in [0x68]:  # IMU
            device_info['potential_uses'] = [
                'Orientation sensing',
                'Acceleration measurement',
                'Gyroscope data'
            ]
        
        return device_info
    
    def _scan_gpio_capabilities(self):
        """Test GPIO pin capabilities."""
        
        print("📍 GPIO Capabilities")
        print("-" * 20)
        
        if not GPIO_AVAILABLE:
            print("❌ RPi.GPIO not available - install with: pip3 install RPi.GPIO")
            self.discovered['gpio_capabilities'] = {'error': 'gpio_not_available'}
            return
        
        try:
            # Set up GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Test common robotics GPIO pins
            test_pins = [18, 19, 20, 21, 12, 13, 16, 26]
            gpio_info = {
                'available_pins': [],
                'pwm_capable_pins': [],
                'used_pins': []
            }
            
            for pin in test_pins:
                try:
                    # Test if pin is available
                    GPIO.setup(pin, GPIO.OUT)
                    gpio_info['available_pins'].append(pin)
                    
                    # Test PWM capability
                    try:
                        pwm = GPIO.PWM(pin, 1000)  # 1kHz test
                        pwm.start(0)
                        time.sleep(0.001)  # Brief test
                        pwm.stop()
                        gpio_info['pwm_capable_pins'].append(pin)
                        
                        if self.detailed:
                            print(f"✅ GPIO {pin}: Available, PWM capable")
                    except:
                        if self.detailed:
                            print(f"✅ GPIO {pin}: Available, no PWM")
                    
                    GPIO.cleanup(pin)
                    
                except Exception as e:
                    gpio_info['used_pins'].append(pin)
                    if self.detailed:
                        print(f"❌ GPIO {pin}: In use or unavailable")
            
            self.discovered['gpio_capabilities'] = gpio_info
            
            print(f"✅ Available GPIO pins: {len(gpio_info['available_pins'])}")
            print(f"✅ PWM capable pins: {len(gpio_info['pwm_capable_pins'])}")
            if gpio_info['pwm_capable_pins']:
                print(f"   PWM pins: {gpio_info['pwm_capable_pins']}")
            
        except Exception as e:
            print(f"❌ GPIO scan failed: {e}")
            self.discovered['gpio_capabilities'] = {'error': str(e)}
        finally:
            try:
                GPIO.cleanup()
            except:
                pass
        
        print()
    
    def _scan_spi_interfaces(self):
        """Check SPI interface availability."""
        
        print("🔄 SPI Interfaces")
        print("-" * 20)
        
        if not SPI_AVAILABLE:
            print("❌ spidev not available - install with: pip3 install spidev")
            self.discovered['spi_devices'] = {'error': 'spi_not_available'}
            return
        
        try:
            spi_info = {
                'buses_available': [],
                'devices_tested': []
            }
            
            # Test common SPI buses
            for bus in [0, 1]:
                for device in [0, 1]:
                    try:
                        spi = spidev.SpiDev()
                        spi.open(bus, device)
                        spi.max_speed_hz = 1000000
                        
                        # Test basic communication
                        response = spi.xfer([0x00])
                        spi.close()
                        
                        spi_info['buses_available'].append(f"SPI{bus}.{device}")
                        if self.detailed:
                            print(f"✅ SPI{bus}.{device}: Available")
                        
                    except Exception:
                        # SPI device not available
                        continue
            
            self.discovered['spi_devices'] = spi_info
            
            if spi_info['buses_available']:
                print(f"✅ Available SPI buses: {', '.join(spi_info['buses_available'])}")
            else:
                print("⚠️  No SPI devices detected (may be disabled)")
            
        except Exception as e:
            print(f"❌ SPI scan failed: {e}")
            self.discovered['spi_devices'] = {'error': str(e)}
        
        print()
    
    def _test_sunfounder_integration(self):
        """Test SunFounder API availability and capabilities."""
        
        print("🤖 SunFounder Integration")
        print("-" * 25)
        
        if not SUNFOUNDER_AVAILABLE:
            print("❌ PiCar-X library not available")
            print("   Install with: cd ~/picar-x && sudo python3 setup.py install")
            self.discovered['sunfounder_status'] = {'error': 'picarx_not_available'}
            return
        
        try:
            # Test PiCar-X initialization
            px = Picarx()
            
            sunfounder_info = {
                'picarx_available': True,
                'api_methods': [],
                'hardware_interfaces': {}
            }
            
            # Discover available methods
            methods = [attr for attr in dir(px) if not attr.startswith('_')]
            sunfounder_info['api_methods'] = methods
            
            # Test specific capabilities
            capabilities = {
                'motor_control': hasattr(px, 'forward') and hasattr(px, 'backward'),
                'steering_control': hasattr(px, 'set_dir_servo_angle'),
                'camera_control': hasattr(px, 'set_camera_servo1_angle'),
                'sensor_access': hasattr(px, 'ultrasonic')
            }
            
            sunfounder_info['capabilities'] = capabilities
            
            if self.detailed:
                print("Available PiCar-X methods:")
                for method in sorted(methods):
                    print(f"  • px.{method}()")
            
            print("✅ SunFounder PiCar-X API available")
            for capability, available in capabilities.items():
                status = "✅" if available else "❌"
                print(f"   {status} {capability.replace('_', ' ').title()}")
            
            self.discovered['sunfounder_status'] = sunfounder_info
            
        except Exception as e:
            print(f"❌ SunFounder test failed: {e}")
            self.discovered['sunfounder_status'] = {'error': str(e)}
        
        print()
    
    def _generate_recommendations(self):
        """Generate recommendations for hardware usage."""
        
        print("💡 Recommendations")
        print("-" * 20)
        
        recommendations = []
        
        # SunFounder recommendations
        if self.discovered['sunfounder_status'].get('picarx_available'):
            recommendations.append({
                'category': 'Primary Control',
                'recommendation': 'Use SunFounder API for main robot control',
                'reason': 'Proven, safe, well-documented interface',
                'implementation': 'px.forward(), px.set_dir_servo_angle()'
            })
        
        # I2C recommendations
        i2c_devices = self.discovered['i2c_devices'].get('devices', [])
        robot_hat_found = any(d['address'] == 0x14 for d in i2c_devices)
        
        if robot_hat_found:
            recommendations.append({
                'category': 'Granular Control',
                'recommendation': 'Access Robot HAT directly for motor current sensing',
                'reason': 'Enable proprioceptive feedback for motor learning',
                'implementation': 'smbus I2C communication to 0x14'
            })
        
        # GPIO recommendations
        gpio_info = self.discovered['gpio_capabilities']
        pwm_pins = gpio_info.get('pwm_capable_pins', [])
        
        if len(pwm_pins) >= 2:
            recommendations.append({
                'category': 'Individual Motor Control',
                'recommendation': 'Use GPIO PWM for independent wheel control',
                'reason': 'Enable differential steering learning',
                'implementation': f'GPIO.PWM on pins {pwm_pins[:2]}'
            })
        
        # Integration recommendation
        if len(recommendations) > 1:
            recommendations.append({
                'category': 'Hybrid Approach',
                'recommendation': 'Layer granular control over SunFounder API',
                'reason': 'Combine safety with learning opportunities',
                'implementation': 'Use SunFounder for safety, GPIO/I2C for experiments'
            })
        
        self.discovered['recommendations'] = recommendations
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['category']}: {rec['recommendation']}")
            if self.detailed:
                print(f"   Reason: {rec['reason']}")
                print(f"   Implementation: {rec['implementation']}")
        
        print()
    
    def print_summary(self):
        """Print a concise summary of findings."""
        
        print("📊 DISCOVERY SUMMARY")
        print("=" * 50)
        
        # Quick stats
        i2c_count = len(self.discovered['i2c_devices'].get('devices', []))
        gpio_count = len(self.discovered['gpio_capabilities'].get('available_pins', []))
        pwm_count = len(self.discovered['gpio_capabilities'].get('pwm_capable_pins', []))
        sunfounder_ok = self.discovered['sunfounder_status'].get('picarx_available', False)
        
        print(f"I2C devices found: {i2c_count}")
        print(f"Available GPIO pins: {gpio_count}")
        print(f"PWM capable pins: {pwm_count}")
        print(f"SunFounder API: {'✅ Available' if sunfounder_ok else '❌ Not available'}")
        print(f"Recommendations: {len(self.discovered['recommendations'])}")
        
        print("\n🎯 Best approach:")
        if self.discovered['recommendations']:
            primary_rec = self.discovered['recommendations'][0]
            print(f"   {primary_rec['recommendation']}")
        else:
            print("   Install SunFounder drivers first")


def main():
    """Main discovery script."""
    
    parser = argparse.ArgumentParser(description='Discover hardware interfaces on PiCar-X')
    parser.add_argument('--detailed', action='store_true', help='Detailed output')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--save', type=str, help='Save results to file (auto-detects format from extension)')
    args = parser.parse_args()
    
    # Run discovery
    discovery = HardwareDiscovery(detailed=args.detailed)
    results = discovery.scan_all_interfaces()
    
    if args.save:
        # Save to file with auto-format detection
        if args.save.endswith('.json'):
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {args.save} (JSON format)")
        else:
            # Save as human-readable text
            with open(args.save, 'w') as f:
                # Redirect print output to file
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                
                discovery.print_summary()
                print("\n" + "="*50)
                print("DETAILED HARDWARE DISCOVERY RESULTS")
                print("="*50)
                
                # Print detailed results
                if results['i2c_devices'].get('devices'):
                    print(f"\nI2C DEVICES ({len(results['i2c_devices']['devices'])} found):")
                    for device in results['i2c_devices']['devices']:
                        print(f"  • 0x{device['address']:02X}: {device['description']}")
                        for use in device.get('potential_uses', []):
                            print(f"    - {use}")
                
                if results['gpio_capabilities'].get('pwm_capable_pins'):
                    pins = results['gpio_capabilities']['pwm_capable_pins']
                    print(f"\nPWM CAPABLE GPIO PINS ({len(pins)} found):")
                    for pin in pins:
                        print(f"  • GPIO {pin}")
                
                if results['sunfounder_status'].get('capabilities'):
                    caps = results['sunfounder_status']['capabilities']
                    print(f"\nSUNFOUNDER CAPABILITIES:")
                    for cap, available in caps.items():
                        status = "✅" if available else "❌"
                        print(f"  {status} {cap.replace('_', ' ').title()}")
                
                if results['recommendations']:
                    print(f"\nRECOMMENDATIONS:")
                    for i, rec in enumerate(results['recommendations'], 1):
                        print(f"\n{i}. {rec['category']}: {rec['recommendation']}")
                        print(f"   Reason: {rec['reason']}")
                        print(f"   Implementation: {rec['implementation']}")
                
                # Restore stdout
                sys.stdout = old_stdout
            
            print(f"💾 Results saved to {args.save} (text format)")
    
    elif args.json:
        # Output JSON for programmatic use
        print(json.dumps(results, indent=2))
    else:
        # Human-readable summary
        discovery.print_summary()
        
        print("\n💾 Save options:")
        print("   python3 hardware_discovery.py --save hardware_report.txt")
        print("   python3 hardware_discovery.py --save hardware_data.json")


if __name__ == "__main__":
    main()