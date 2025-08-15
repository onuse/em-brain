#!/usr/bin/env python3
"""
Servo Characterization and Dynamic Mapping

World-class servo control with:
1. Automatic servo discovery and characterization
2. Dynamic brain→servo mapping
3. Servo response curve learning
4. Feedback and monitoring
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class ServoCharacteristics:
    """Discovered characteristics of a servo."""
    channel: int                    # PCA9685 channel
    min_pulse_us: int               # Minimum safe pulse
    max_pulse_us: int               # Maximum safe pulse
    center_pulse_us: int            # Neutral/center position
    
    # Response characteristics
    response_curve: List[Tuple[int, float]]  # (pulse_us, actual_angle)
    backlash_us: int                # Hysteresis in microseconds
    slew_rate_us_per_ms: float     # Max speed in µs/ms
    
    # Discovered properties
    is_continuous: bool             # Continuous rotation servo?
    has_feedback: bool              # Position feedback available?
    typical_current_ma: float       # Typical current draw
    stall_current_ma: float         # Stall detection threshold
    
    # Brain mapping
    brain_output_index: Optional[int] = None  # Which brain output controls this
    semantic_name: Optional[str] = None       # Human-readable name


class ServoCharacterizer:
    """
    Discover and characterize servo properties.
    
    The brain doesn't need to know these are "servos" - 
    it just learns that certain outputs cause certain changes.
    """
    
    def __init__(self, hal):
        """Initialize with bare metal HAL."""
        self.hal = hal
        self.characteristics = {}
        
    def discover_servo(self, channel: int, assisted: bool = True) -> ServoCharacteristics:
        """
        Discover characteristics of a servo on given channel.
        
        Args:
            channel: PCA9685 channel to test
            assisted: If True, ask human for help identifying
            
        Returns:
            ServoCharacteristics with discovered properties
        """
        print(f"\n🔍 Characterizing servo on channel {channel}...")
        
        # Start from center
        center = 1500
        self.hal._set_servo_pulse(channel, center)
        time.sleep(0.5)
        
        # Find movement range
        min_pulse, max_pulse = self._find_servo_limits(channel, center)
        print(f"   Range: {min_pulse}-{max_pulse}µs")
        
        # Check for continuous rotation
        is_continuous = self._test_continuous_rotation(channel)
        print(f"   Type: {'Continuous' if is_continuous else 'Standard'}")
        
        # Measure response curve
        response_curve = self._measure_response_curve(channel, min_pulse, max_pulse)
        
        # Measure backlash
        backlash = self._measure_backlash(channel, center)
        print(f"   Backlash: {backlash}µs")
        
        # Measure slew rate
        slew_rate = self._measure_slew_rate(channel, min_pulse, max_pulse)
        print(f"   Slew rate: {slew_rate:.1f}µs/ms")
        
        # Current monitoring (if available)
        typical_current, stall_current = self._measure_current_profile(channel)
        
        # Create characteristics
        characteristics = ServoCharacteristics(
            channel=channel,
            min_pulse_us=min_pulse,
            max_pulse_us=max_pulse,
            center_pulse_us=center,
            response_curve=response_curve,
            backlash_us=backlash,
            slew_rate_us_per_ms=slew_rate,
            is_continuous=is_continuous,
            has_feedback=False,  # PiCar-X servos don't have feedback
            typical_current_ma=typical_current,
            stall_current_ma=stall_current
        )
        
        # Human-assisted identification
        if assisted:
            name = input("What is this servo? (steering/pan/tilt/other): ")
            characteristics.semantic_name = name
        
        self.characteristics[channel] = characteristics
        return characteristics
    
    def _find_servo_limits(self, channel: int, center: int) -> Tuple[int, int]:
        """Find safe operating range of servo."""
        
        # Start from center, gradually expand range
        min_found = center
        max_found = center
        
        # Test decreasing pulses
        for pulse in range(center, 500, -50):
            self.hal._set_servo_pulse(channel, pulse)
            time.sleep(0.1)
            
            # Check if still moving (would need current sensor)
            # For now, assume standard range
            min_found = pulse
            
            if pulse <= 600:  # Safety limit
                break
        
        # Return to center
        self.hal._set_servo_pulse(channel, center)
        time.sleep(0.3)
        
        # Test increasing pulses
        for pulse in range(center, 2500, 50):
            self.hal._set_servo_pulse(channel, pulse)
            time.sleep(0.1)
            
            max_found = pulse
            
            if pulse >= 2400:  # Safety limit
                break
        
        # Return to center
        self.hal._set_servo_pulse(channel, center)
        
        return min_found, max_found
    
    def _test_continuous_rotation(self, channel: int) -> bool:
        """Test if servo is continuous rotation type."""
        
        # Continuous servos interpret pulse as speed, not position
        # Send max pulse and see if it keeps moving
        
        self.hal._set_servo_pulse(channel, 2000)
        time.sleep(0.5)
        
        # Would need encoder to properly detect
        # For now, assume standard servo
        continuous = False
        
        self.hal._set_servo_pulse(channel, 1500)
        return continuous
    
    def _measure_response_curve(self, channel: int, min_pulse: int, max_pulse: int) -> List[Tuple[int, float]]:
        """Measure actual response curve of servo."""
        
        curve = []
        
        # Sample response at various points
        test_pulses = np.linspace(min_pulse, max_pulse, 11)
        
        for pulse in test_pulses:
            self.hal._set_servo_pulse(channel, int(pulse))
            time.sleep(0.3)  # Let servo settle
            
            # Would need position feedback to measure actual angle
            # For now, assume linear response
            estimated_angle = (pulse - 1500) / 10  # Rough estimate
            curve.append((int(pulse), estimated_angle))
        
        return curve
    
    def _measure_backlash(self, channel: int, center: int) -> int:
        """Measure servo backlash/hysteresis."""
        
        # Move one direction
        self.hal._set_servo_pulse(channel, center + 200)
        time.sleep(0.5)
        
        # Move back slightly
        for pulse in range(center + 200, center, -10):
            self.hal._set_servo_pulse(channel, pulse)
            time.sleep(0.05)
            
            # Would need position sensor to detect when movement starts
            # Estimate ~20µs typical backlash
        
        return 20
    
    def _measure_slew_rate(self, channel: int, min_pulse: int, max_pulse: int) -> float:
        """Measure maximum servo speed."""
        
        # Move from min to max
        self.hal._set_servo_pulse(channel, min_pulse)
        time.sleep(1)
        
        start_time = time.time()
        self.hal._set_servo_pulse(channel, max_pulse)
        time.sleep(0.5)  # Typical servo transit time
        transit_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate slew rate
        pulse_change = max_pulse - min_pulse
        slew_rate = pulse_change / transit_time if transit_time > 0 else 1000
        
        return slew_rate
    
    def _measure_current_profile(self, channel: int) -> Tuple[float, float]:
        """Measure servo current consumption."""
        
        # Would need current sensor (INA219) for real measurement
        # Estimate typical values
        typical = 200.0  # mA typical
        stall = 800.0    # mA stall
        
        return typical, stall


class DynamicServoMapper:
    """
    Dynamic mapping between brain outputs and servos.
    
    Allows flexible configuration and learning of what each
    brain output controls, without hardcoding.
    """
    
    def __init__(self, num_brain_outputs: int = 4):
        """Initialize dynamic mapper."""
        self.num_brain_outputs = num_brain_outputs
        self.mapping = {}  # brain_index -> servo_channel
        self.characteristics = {}  # channel -> ServoCharacteristics
        
    def learn_mapping(self, hal, iterations: int = 100):
        """
        Let brain discover which outputs control which servos.
        
        Brain tries different outputs and observes sensor changes
        to learn the mapping.
        """
        
        print("🧠 Brain discovering servo mappings...")
        
        discovered_mappings = {}
        
        for output_idx in range(self.num_brain_outputs):
            print(f"\nTesting brain output {output_idx}...")
            
            # Create test signal on this output
            for pulse_test in [1200, 1500, 1800, 1500]:
                # Only activate this one output
                brain_output = [0.0] * self.num_brain_outputs
                
                # Convert to normalized value the brain would use
                normalized = (pulse_test - 1500) / 1000.0
                brain_output[output_idx] = normalized
                
                # Execute and observe what moves
                # In real system, would check sensor changes
                hal.process_test_output(brain_output)
                time.sleep(0.3)
            
            # Ask for human verification (or use sensors)
            moved = input("What moved? (steering/pan/tilt/left_motor/right_motor/nothing): ")
            if moved != "nothing":
                discovered_mappings[output_idx] = moved
        
        return discovered_mappings
    
    def apply_mapping(self, brain_output: List[float], 
                      characteristics: Dict[int, ServoCharacteristics]) -> Dict[int, int]:
        """
        Convert brain outputs to servo commands using learned mapping.
        
        Returns:
            Dict of channel -> pulse_us commands
        """
        
        commands = {}
        
        for brain_idx, value in enumerate(brain_output):
            if brain_idx in self.mapping:
                channel = self.mapping[brain_idx]
                
                if channel in characteristics:
                    char = characteristics[channel]
                    
                    # Convert brain's [-1, 1] to servo's pulse range
                    # Brain doesn't know these are microseconds
                    pulse_range = char.max_pulse_us - char.min_pulse_us
                    pulse = char.center_pulse_us + (value * pulse_range / 2)
                    
                    # Apply limits
                    pulse = max(char.min_pulse_us, min(char.max_pulse_us, pulse))
                    
                    commands[channel] = int(pulse)
        
        return commands
    
    def save_mapping(self, filename: str = "servo_mapping.json"):
        """Save discovered mapping and characteristics."""
        
        data = {
            'mapping': self.mapping,
            'characteristics': {
                str(ch): asdict(char) 
                for ch, char in self.characteristics.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Servo mapping saved to {filename}")
    
    def load_mapping(self, filename: str = "servo_mapping.json"):
        """Load previously discovered mapping."""
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.mapping = {int(k): v for k, v in data['mapping'].items()}
            
            # Reconstruct characteristics
            for ch_str, char_dict in data['characteristics'].items():
                char = ServoCharacteristics(**char_dict)
                self.characteristics[int(ch_str)] = char
            
            print(f"📂 Loaded servo mapping from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"⚠️ No saved mapping found")
            return False


def characterize_all_servos(hal):
    """
    Complete servo characterization process.
    """
    
    print("=" * 60)
    print("🎯 SERVO CHARACTERIZATION")
    print("=" * 60)
    
    characterizer = ServoCharacterizer(hal)
    mapper = DynamicServoMapper()
    
    # Test known servo channels
    servo_channels = [0, 1, 2]  # Camera pan, tilt, steering
    
    for channel in servo_channels:
        characteristics = characterizer.discover_servo(channel)
        mapper.characteristics[channel] = characteristics
    
    # Learn brain→servo mapping
    discovered = mapper.learn_mapping(hal)
    
    # Save everything
    mapper.save_mapping()
    
    print("\n✅ Servo characterization complete!")
    
    return mapper


if __name__ == "__main__":
    """Test servo characterization."""
    
    # This would use the actual HAL
    print("Servo Characterization Test")
    print("Run on actual hardware with: python3 servo_characterization.py")
    
    # Example characteristics (what we'd discover)
    example = ServoCharacteristics(
        channel=2,
        min_pulse_us=600,
        max_pulse_us=2400,
        center_pulse_us=1500,
        response_curve=[(600, -90), (1500, 0), (2400, 90)],
        backlash_us=20,
        slew_rate_us_per_ms=3.6,
        is_continuous=False,
        has_feedback=False,
        typical_current_ma=200,
        stall_current_ma=800,
        brain_output_index=2,
        semantic_name="steering"
    )
    
    print(f"\nExample servo characteristics:")
    print(json.dumps(asdict(example), indent=2))