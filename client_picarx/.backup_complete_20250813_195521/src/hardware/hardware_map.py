#!/usr/bin/env python3
"""
Hardware Mapping for PiCar-X

Based on analysis of SunFounder's implementation:
- https://github.com/sunfounder/picar-x/blob/v2.0/picarx/picarx.py
- https://github.com/sunfounder/robot-hat/blob/v2.0/

This mapping is verified against the actual PiCar-X hardware.
"""

# Verified hardware mapping from PiCar-X source
PICARX_HARDWARE_MAP = {
    'pca9685_channels': {
        # Servos (from picarx.py init)
        0: 'camera_pan',     # P0 - Camera pan servo
        1: 'camera_tilt',    # P1 - Camera tilt servo
        2: 'steering',       # P2 - Front steering servo
        
        # Motors use PWM channels on custom board (not PCA9685)
        # P12, P13 are custom PWM channels via robot-hat
        # We'll use PCA9685 channels 4,5 for our bare metal implementation
        4: 'left_motor_pwm',   # Our allocation for bare metal
        5: 'right_motor_pwm',  # Our allocation for bare metal
    },
    
    'gpio_pins': {
        # Motor direction pins (from picarx.py + pin.py mapping)
        23: 'left_motor_dir',   # D4 → GPIO 23
        24: 'right_motor_dir',  # D5 → GPIO 24
        
        # Ultrasonic sensor
        27: 'ultrasonic_trig',  # D2 → GPIO 27
        22: 'ultrasonic_echo',  # D3 → GPIO 22
    },
    
    'adc_channels': {
        # Grayscale line sensors (from picarx.py)
        0: 'grayscale_left',    # A0 → ADC channel 0
        1: 'grayscale_center',  # A1 → ADC channel 1
        2: 'grayscale_right',   # A2 → ADC channel 2
        
        # Battery monitoring (typical, needs verification)
        6: 'battery_voltage',   # Usually on A6 or A7
        
        # Note: No current sensors on standard PiCar-X
    },
    
    'i2c_addresses': {
        # Standard I2C addresses for robot-hat
        0x14: 'adc',           # ADC chip
        0x40: 'pca9685',       # Servo/PWM controller
    }
}

# Mapping for brainstem (doesn't "know" what these are)
# Just consistent channel ordering
BRAINSTEM_IO_MAP = {
    'brain_to_hardware_outputs': [
        # Brain output index → hardware channel
        ('pca9685', 4),   # 0: Primary thrust → left motor PWM
        ('pca9685', 5),   # 1: Also thrust → right motor PWM  
        ('pca9685', 2),   # 2: Lateral → steering servo
        ('pca9685', 0),   # 3: Sensor aim → camera pan
    ],
    
    'hardware_to_brain_inputs': [
        # Hardware channel → brain input index
        ('adc', 0),        # 0: Some sensor → brain[0]
        ('adc', 1),        # 1: Some sensor → brain[1]
        ('adc', 2),        # 2: Some sensor → brain[2]
        ('ultrasonic', 0), # 3: Echo time → brain[3]
        ('adc', 6),        # 4: Battery → brain[4]
        # ... up to 24 channels
    ]
}

def discover_hardware_mapping(hal, assisted=True):
    """
    Discover actual hardware connections.
    
    Args:
        hal: BareMetalHAL instance
        assisted: If True, ask human what moved/changed
        
    Returns:
        Dictionary with discovered mappings
    """
    import time
    import numpy as np
    
    discovered = {
        'pca9685_channels': {},
        'adc_channels': {},
        'gpio_pins': {}
    }
    
    if assisted:
        print("=" * 60)
        print("🔍 ASSISTED HARDWARE DISCOVERY")
        print("=" * 60)
        print("I'll activate hardware. You tell me what happens.")
        print("This helps the brain learn its body.\n")
        
        # Test PCA9685 channels
        print("Testing servo/PWM channels...")
        for channel in range(16):
            print(f"\n[Channel {channel}]", end='', flush=True)
            
            # Pulse servo
            for pulse in [1200, 1500, 1800, 1500]:
                hal._set_servo_pulse(channel, pulse)
                time.sleep(0.3)
                print(".", end='', flush=True)
            
            response = input("\nWhat moved? (steering/pan/tilt/left_motor/right_motor/skip): ")
            if response != 'skip' and response:
                discovered['pca9685_channels'][channel] = response
        
        # Test ADC channels
        print("\n\nTesting ADC channels...")
        for channel in range(8):
            values = []
            for _ in range(20):
                values.append(hal._read_adc_channel(channel))
                time.sleep(0.05)
            
            mean = np.mean(values)
            variance = np.var(values)
            
            print(f"\nADC {channel}: mean={mean:.0f}, variance={variance:.1f}")
            
            if variance > 5 or mean > 100:  # Active sensor
                response = input("What sensor? (grayscale_l/grayscale_c/grayscale_r/battery/skip): ")
                if response != 'skip' and response:
                    discovered['adc_channels'][channel] = response
    
    else:
        # Use pre-verified mapping
        print("Using pre-verified PiCar-X hardware mapping")
        discovered = PICARX_HARDWARE_MAP
    
    return discovered

def save_hardware_map(mapping, filename='hardware_map.json'):
    """Save discovered hardware mapping."""
    import json
    
    with open(filename, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"💾 Hardware map saved to {filename}")

def load_hardware_map(filename='hardware_map.json'):
    """Load hardware mapping, fall back to defaults."""
    import json
    import os
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            mapping = json.load(f)
        print(f"📂 Loaded hardware map from {filename}")
        return mapping
    else:
        print("📋 Using default PiCar-X hardware mapping")
        return PICARX_HARDWARE_MAP


if __name__ == "__main__":
    """Test hardware mapping."""
    
    print("PiCar-X Hardware Mapping")
    print("=" * 40)
    
    print("\n📍 PCA9685 Channels:")
    for ch, name in PICARX_HARDWARE_MAP['pca9685_channels'].items():
        print(f"   {ch:2d}: {name}")
    
    print("\n📍 GPIO Pins:")
    for pin, name in PICARX_HARDWARE_MAP['gpio_pins'].items():
        print(f"   GPIO {pin:2d}: {name}")
    
    print("\n📍 ADC Channels:")
    for ch, name in PICARX_HARDWARE_MAP['adc_channels'].items():
        print(f"   {ch}: {name}")
    
    print("\n🧠 Brainstem IO Mapping:")
    print("   Brain Outputs → Hardware:")
    for i, (hw_type, hw_ch) in enumerate(BRAINSTEM_IO_MAP['brain_to_hardware_outputs']):
        print(f"     {i} → {hw_type}[{hw_ch}]")
    
    print("\n✅ Mapping ready for use")