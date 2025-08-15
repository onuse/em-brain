#!/usr/bin/env python3
"""
Simple, Honest Servo Control

No discovery nonsense. Just hardcoded mappings and simple calibration.
The brain learns what outputs do through sensor feedback, not servo feedback.
"""

import json
import numpy as np
from typing import List, Dict, Tuple


class SimpleServoControl:
    """
    Pragmatic servo control with hardcoded mappings.
    
    We know:
    - Which channels have which servos (fixed by PCB)
    - Safe operating ranges (standard servo specs)
    - Brain outputs map to servos (we decide this)
    
    We calibrate:
    - Center points (might not be exactly 1500µs)
    - Actual range (might be less than full range)
    
    Brain learns:
    - What each output does (through sensor feedback)
    - How to compensate for physics
    """
    
    # Fixed by hardware design - no discovery needed
    SERVO_CHANNELS = {
        'steering': 2,      # PCA9685 channel 2
        'camera_pan': 0,    # PCA9685 channel 0  
        'camera_tilt': 1,   # PCA9685 channel 1
    }
    
    # Safe defaults for standard servos
    DEFAULT_LIMITS = {
        'steering': {
            'min': 1000,
            'max': 2000,
            'center': 1500,
        },
        'camera_pan': {
            'min': 1000,
            'max': 2000,
            'center': 1500,
        },
        'camera_tilt': {
            'min': 1000,
            'max': 2000,
            'center': 1500,
        }
    }
    
    # Fixed brain output mapping (we decide this)
    BRAIN_OUTPUT_MAP = [
        None,           # Output 0: Used for motors (forward/back)
        None,           # Output 1: Used for motors (differential)
        'steering',     # Output 2: Steering servo
        'camera_pan',   # Output 3: Camera pan (or tilt, configurable)
    ]
    
    def __init__(self, hal):
        """Initialize with HAL and load calibration if exists."""
        self.hal = hal
        
        # Start with defaults
        self.limits = self.DEFAULT_LIMITS.copy()
        
        # Load calibration if available
        self.load_calibration()
        
        # Initialize all servos to center
        self.center_all_servos()
        
        print("🎯 Simple servo control initialized")
        print(f"   Steering: ch{self.SERVO_CHANNELS['steering']}")
        print(f"   Camera: ch{self.SERVO_CHANNELS['camera_pan']}, "
              f"ch{self.SERVO_CHANNELS['camera_tilt']}")
    
    def calibrate_servo(self, servo_name: str):
        """
        Simple calibration for one servo.
        Human operates it to find good limits.
        """
        if servo_name not in self.SERVO_CHANNELS:
            print(f"❌ Unknown servo: {servo_name}")
            return
            
        channel = self.SERVO_CHANNELS[servo_name]
        current = self.limits[servo_name]
        
        print(f"\n🔧 Calibrating {servo_name} (channel {channel})")
        print(f"Current: {current['min']}-{current['center']}-{current['max']}µs")
        
        # Test current center
        self.hal._set_servo_pulse(channel, current['center'])
        new_center = input(f"New center (Enter for {current['center']}): ")
        if new_center:
            current['center'] = int(new_center)
            self.hal._set_servo_pulse(channel, current['center'])
        
        # Test minimum
        print("\nTesting minimum position...")
        test_min = current['center'] - 300
        self.hal._set_servo_pulse(channel, test_min)
        new_min = input(f"New minimum (Enter for {current['min']}): ")
        if new_min:
            current['min'] = int(new_min)
        
        # Test maximum  
        print("\nTesting maximum position...")
        test_max = current['center'] + 300
        self.hal._set_servo_pulse(channel, test_max)
        new_max = input(f"New maximum (Enter for {current['max']}): ")
        if new_max:
            current['max'] = int(new_max)
        
        # Return to center
        self.hal._set_servo_pulse(channel, current['center'])
        
        print(f"✅ Calibrated: {current['min']}-{current['center']}-{current['max']}µs")
        
        # Save calibration
        self.save_calibration()
    
    def process_brain_output(self, brain_output: List[float]) -> Dict[int, int]:
        """
        Convert brain outputs to servo commands.
        
        Simple, direct, no physics simulation.
        Brain learns through sensor feedback.
        
        Args:
            brain_output: List of 4 values [-1, 1]
            
        Returns:
            Dict of channel -> pulse_us commands
        """
        commands = {}
        
        for i, servo_name in enumerate(self.BRAIN_OUTPUT_MAP):
            if servo_name and i < len(brain_output):
                channel = self.SERVO_CHANNELS[servo_name]
                limits = self.limits[servo_name]
                
                # Convert [-1, 1] to pulse microseconds
                # -1 = min, 0 = center, 1 = max
                value = brain_output[i]
                
                if value < 0:
                    # Negative: center to min
                    pulse = limits['center'] + value * (limits['center'] - limits['min'])
                else:
                    # Positive: center to max
                    pulse = limits['center'] + value * (limits['max'] - limits['center'])
                
                # Clamp to limits
                pulse = int(np.clip(pulse, limits['min'], limits['max']))
                
                # Execute
                self.hal._set_servo_pulse(channel, pulse)
                commands[channel] = pulse
        
        return commands
    
    def center_all_servos(self):
        """Move all servos to center position."""
        for servo_name, channel in self.SERVO_CHANNELS.items():
            center = self.limits[servo_name]['center']
            self.hal._set_servo_pulse(channel, center)
        print("✅ All servos centered")
    
    def save_calibration(self, filename: str = "servo_calibration.json"):
        """Save calibration to file."""
        with open(filename, 'w') as f:
            json.dump(self.limits, f, indent=2)
        print(f"💾 Calibration saved to {filename}")
    
    def load_calibration(self, filename: str = "servo_calibration.json"):
        """Load calibration from file."""
        try:
            with open(filename, 'r') as f:
                self.limits = json.load(f)
            print(f"📂 Loaded calibration from {filename}")
            return True
        except FileNotFoundError:
            print("📋 Using default servo limits")
            return False
    
    def get_info(self) -> Dict:
        """Get current servo configuration."""
        return {
            'channels': self.SERVO_CHANNELS,
            'limits': self.limits,
            'brain_mapping': {
                f"output_{i}": servo 
                for i, servo in enumerate(self.BRAIN_OUTPUT_MAP)
                if servo
            }
        }


def calibrate_picarx_servos(hal):
    """
    One-time calibration for PiCar-X servos.
    """
    print("=" * 60)
    print("🔧 PICAR-X SERVO CALIBRATION")
    print("=" * 60)
    print("\nThis will move servos to find their limits.")
    print("Make sure robot is secure!\n")
    
    controller = SimpleServoControl(hal)
    
    # Calibrate each servo
    servos = ['steering', 'camera_pan', 'camera_tilt']
    
    for servo in servos:
        response = input(f"\nCalibrate {servo}? (y/n): ")
        if response.lower() == 'y':
            controller.calibrate_servo(servo)
    
    print("\n✅ Calibration complete!")
    print("\nFinal configuration:")
    print(json.dumps(controller.get_info(), indent=2))
    
    return controller


if __name__ == "__main__":
    """Test servo control."""
    
    print("Simple Servo Control Test")
    
    # Mock HAL for testing
    class MockHAL:
        def _set_servo_pulse(self, channel: int, pulse: int):
            print(f"  CH{channel} → {pulse}µs")
    
    hal = MockHAL()
    controller = SimpleServoControl(hal)
    
    # Test brain outputs
    print("\n🧠 Testing brain outputs:")
    
    test_patterns = [
        ([0, 0, 0, 0], "Center"),
        ([0, 0, 0.5, 0], "Steer right"),
        ([0, 0, -0.5, 0], "Steer left"),
        ([0, 0, 0, 0.5], "Camera pan right"),
        ([0, 0, 0, -0.5], "Camera pan left"),
    ]
    
    for brain_output, description in test_patterns:
        print(f"\n{description}: {brain_output}")
        commands = controller.process_brain_output(brain_output)
    
    print("\n✅ Test complete")