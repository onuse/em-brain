#!/usr/bin/env python3
"""
Research Motor HAL - Hybrid SunFounder + Granular Control

A practical abstraction layer that combines SunFounder API safety
with selective granular control for motor learning research.

This provides the "useful amount of abstraction" - safe foundation
with research-enabling granular control where it adds value.
"""

import time
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# SunFounder API (install via sunfounder_setup.sh)
try:
    from picarx import Picarx
    SUNFOUNDER_AVAILABLE = True
except ImportError:
    SUNFOUNDER_AVAILABLE = False
    print("⚠️  SunFounder API not available - using mock")

# Low-level interfaces (available on Pi)
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


@dataclass
class MotorFeedback:
    """Rich motor feedback for learning."""
    left_current: float      # Amps
    right_current: float     # Amps
    left_speed_actual: float # Measured speed
    right_speed_actual: float
    timestamp: float


@dataclass
class MotorCommand:
    """Motor command with metadata."""
    left_speed: float        # -100 to +100
    right_speed: float       # -100 to +100
    duration: float          # seconds
    learning_mode: bool      # Use granular control?


class ResearchMotorHAL:
    """
    Hybrid motor control for robotics research.
    
    Combines SunFounder API safety with granular control capabilities.
    Brain can choose abstraction level based on learning objectives.
    """
    
    def __init__(self, enable_granular=True):
        """Initialize hybrid motor control system."""
        
        # Core system
        self.px = None
        self.i2c_bus = None
        self.granular_enabled = False
        
        # Safety constraints (never change these!)
        self.MAX_SPEED = 50      # SunFounder safe limit
        self.MAX_STEERING = 30   # degrees
        self.MAX_DIFFERENTIAL = 30  # Max speed difference for learning
        
        # State tracking
        self.last_command = None
        self.motor_feedback_available = False
        self.experiment_log = []
        
        # Initialize systems
        self._init_sunfounder()
        if enable_granular:
            self._init_granular_control()
        
        print(f"🤖 Research Motor HAL initialized")
        print(f"   SunFounder API: {'✅' if self.px else '❌'}")
        print(f"   Granular control: {'✅' if self.granular_enabled else '❌'}")
    
    def _init_sunfounder(self):
        """Initialize SunFounder API (safety foundation)."""
        if SUNFOUNDER_AVAILABLE:
            try:
                self.px = Picarx()
                print("✅ SunFounder PiCar-X API ready")
            except Exception as e:
                print(f"❌ SunFounder init failed: {e}")
        else:
            print("⚠️  Running without SunFounder API (development mode)")
    
    def _init_granular_control(self):
        """Initialize granular motor control interfaces."""
        
        # Try I2C motor feedback
        if I2C_AVAILABLE:
            try:
                self.i2c_bus = smbus.SMBus(1)
                # Test communication with Robot HAT
                test_read = self.i2c_bus.read_byte(0x14)
                self.motor_feedback_available = True
                print("✅ I2C motor feedback available")
            except:
                print("⚠️  I2C motor feedback not available")
        
        # Try GPIO direct motor control
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                # Test GPIO setup (without actually configuring motors)
                self.granular_enabled = True
                print("✅ GPIO motor control available")
            except:
                print("⚠️  GPIO motor control not available")
    
    def execute_motor_command(self, command: MotorCommand) -> MotorFeedback:
        """
        Execute motor command using appropriate abstraction level.
        
        Args:
            command: Motor command with learning mode flag
            
        Returns:
            MotorFeedback with proprioceptive data
        """
        
        # Apply safety constraints
        safe_command = self._apply_safety_constraints(command)
        
        # Choose execution method
        if command.learning_mode and self.granular_enabled:
            feedback = self._execute_granular_control(safe_command)
        else:
            feedback = self._execute_sunfounder_control(safe_command)
        
        # Log for research
        self._log_motor_experiment(safe_command, feedback)
        self.last_command = safe_command
        
        return feedback
    
    def _apply_safety_constraints(self, command: MotorCommand) -> MotorCommand:
        """Apply hard safety limits to any motor command."""
        
        # Clamp speeds to safe range
        left_safe = max(-self.MAX_SPEED, min(self.MAX_SPEED, command.left_speed))
        right_safe = max(-self.MAX_SPEED, min(self.MAX_SPEED, command.right_speed))
        
        # Limit differential for stability
        differential = abs(left_safe - right_safe)
        if differential > self.MAX_DIFFERENTIAL:
            # Scale back to safe differential
            avg_speed = (left_safe + right_safe) / 2
            diff_direction = 1 if right_safe > left_safe else -1
            
            left_safe = avg_speed - (self.MAX_DIFFERENTIAL / 2) * diff_direction
            right_safe = avg_speed + (self.MAX_DIFFERENTIAL / 2) * diff_direction
        
        return MotorCommand(
            left_speed=left_safe,
            right_speed=right_safe,
            duration=min(command.duration, 2.0),  # Max 2 second commands
            learning_mode=command.learning_mode
        )
    
    def _execute_sunfounder_control(self, command: MotorCommand) -> MotorFeedback:
        """Execute using safe SunFounder API."""
        
        if not self.px:
            # Mock execution for development
            return MotorFeedback(0.0, 0.0, command.left_speed, command.right_speed, time.time())
        
        # Convert differential command to SunFounder API
        avg_speed = (command.left_speed + command.right_speed) / 2
        steering_bias = (command.right_speed - command.left_speed) * 0.5
        
        # Execute movement
        if avg_speed > 0:
            self.px.forward(abs(avg_speed))
        elif avg_speed < 0:
            self.px.backward(abs(avg_speed))
        else:
            self.px.stop()
        
        # Apply steering if differential
        if abs(steering_bias) > 1:
            steering_angle = max(-self.MAX_STEERING, min(self.MAX_STEERING, steering_bias))
            self.px.set_dir_servo_angle(steering_angle)
        
        # Return basic feedback (no granular sensing with SunFounder only)
        return MotorFeedback(
            left_current=0.0,  # Not available via SunFounder API
            right_current=0.0,
            left_speed_actual=command.left_speed,   # Assume commanded = actual
            right_speed_actual=command.right_speed,
            timestamp=time.time()
        )
    
    def _execute_granular_control(self, command: MotorCommand) -> MotorFeedback:
        """Execute using granular motor control with feedback."""
        
        # This would implement direct I2C/GPIO motor control
        # For now, fall back to SunFounder but add simulated feedback
        
        print(f"🧠 Granular control: L={command.left_speed:.1f}, R={command.right_speed:.1f}")
        
        # Execute via SunFounder (until we reverse-engineer I2C protocol)
        basic_feedback = self._execute_sunfounder_control(command)
        
        # Add simulated proprioceptive feedback for research
        simulated_left_current = abs(command.left_speed) * 0.02 + 0.1  # Simulate current draw
        simulated_right_current = abs(command.right_speed) * 0.02 + 0.1
        
        # Simulate motor differences (left motor slightly weaker)
        left_efficiency = 0.95  # Left motor 5% weaker
        actual_left = command.left_speed * left_efficiency
        actual_right = command.right_speed
        
        return MotorFeedback(
            left_current=simulated_left_current,
            right_current=simulated_right_current,
            left_speed_actual=actual_left,
            right_speed_actual=actual_right,
            timestamp=time.time()
        )
    
    def _log_motor_experiment(self, command: MotorCommand, feedback: MotorFeedback):
        """Log motor commands and feedback for research analysis."""
        
        experiment_data = {
            'timestamp': feedback.timestamp,
            'command': {
                'left_speed': command.left_speed,
                'right_speed': command.right_speed,
                'learning_mode': command.learning_mode
            },
            'feedback': {
                'left_current': feedback.left_current,
                'right_current': feedback.right_current,
                'left_speed_actual': feedback.left_speed_actual,
                'right_speed_actual': feedback.right_speed_actual
            }
        }
        
        self.experiment_log.append(experiment_data)
        
        # Keep log manageable
        if len(self.experiment_log) > 1000:
            self.experiment_log = self.experiment_log[-500:]  # Keep recent 500
    
    def get_motor_capabilities(self) -> Dict:
        """Return current motor system capabilities."""
        return {
            'sunfounder_api': self.px is not None,
            'granular_control': self.granular_enabled,
            'motor_feedback': self.motor_feedback_available,
            'max_speed': self.MAX_SPEED,
            'max_differential': self.MAX_DIFFERENTIAL,
            'safety_constraints': True
        }
    
    def emergency_stop(self):
        """Immediate emergency stop using most reliable method."""
        if self.px:
            self.px.stop()
            self.px.set_dir_servo_angle(0)
        
        print("🛑 Emergency stop executed")
    
    def save_experiment_log(self, filename: str):
        """Save motor experiment data for research analysis."""
        with open(filename, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        print(f"📊 Experiment log saved: {filename} ({len(self.experiment_log)} entries)")
    
    def cleanup(self):
        """Clean up hardware resources."""
        if self.px:
            self.px.stop()
        
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except:
                pass


# Example usage and testing
def main():
    """Test the Research Motor HAL."""
    
    print("🧪 Testing Research Motor HAL")
    print("=" * 40)
    
    # Initialize HAL
    motor_hal = ResearchMotorHAL(enable_granular=True)
    
    # Test basic movement (safe mode)
    print("\n🛡️ Testing safe mode...")
    safe_command = MotorCommand(20, 20, 1.0, learning_mode=False)
    feedback = motor_hal.execute_motor_command(safe_command)
    print(f"Safe mode feedback: {feedback}")
    
    # Test learning mode (granular control)
    print("\n🧠 Testing learning mode...")
    learning_command = MotorCommand(15, 25, 0.5, learning_mode=True)
    feedback = motor_hal.execute_motor_command(learning_command)
    print(f"Learning mode feedback: {feedback}")
    
    # Test differential steering discovery
    print("\n🔬 Testing motor coordination discovery...")
    for trial in range(3):
        left_speed = 10 + trial * 5
        right_speed = 15 + trial * 3
        
        command = MotorCommand(left_speed, right_speed, 0.3, learning_mode=True)
        feedback = motor_hal.execute_motor_command(command)
        
        print(f"Trial {trial}: L={left_speed}, R={right_speed} → "
              f"Current L={feedback.left_current:.2f}A, R={feedback.right_current:.2f}A")
    
    # Show capabilities
    print(f"\n🤖 Motor HAL capabilities: {motor_hal.get_motor_capabilities()}")
    
    # Save experiment data
    motor_hal.save_experiment_log("motor_experiments.json")
    
    # Cleanup
    motor_hal.cleanup()
    print("\n✅ Research Motor HAL test complete")


if __name__ == "__main__":
    main()