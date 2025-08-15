# Hybrid Control Design: SunFounder + Granular Interfaces

Design document for combining SunFounder APIs with selective lower-level control for motor learning research.

## 🎯 Design Philosophy

**Foundation:** SunFounder API (safe, proven, documented)  
**Enhancement:** Selective granular control where it enables meaningful learning  
**Goal:** Motor coordination research without sacrificing safety or reliability

## 🏗️ Layered Architecture

### Layer 1: Safety Foundation (SunFounder API)
```python
from picarx import Picarx

class SafetyLayer:
    """Always-available safety controls using SunFounder API."""
    
    def __init__(self):
        self.px = Picarx()
        self.emergency_active = False
    
    def emergency_stop(self):
        """Guaranteed safe stop using SunFounder API."""
        self.px.stop()
        self.px.set_dir_servo_angle(0)  # Center steering
        self.emergency_active = True
    
    def safe_forward(self, speed):
        """Safety-limited movement."""
        safe_speed = max(0, min(50, speed))  # Limit to 50 max
        self.px.forward(safe_speed)
    
    def safe_steering(self, angle):
        """Safety-limited steering."""
        safe_angle = max(-30, min(30, angle))  # Limit to ±30°
        self.px.set_dir_servo_angle(safe_angle)
```

### Layer 2: Enhanced Control (Selective Granular)
```python
import smbus
import RPi.GPIO as GPIO

class EnhancedMotorControl:
    """Granular motor control for research, built on safety foundation."""
    
    def __init__(self, safety_layer):
        self.safety = safety_layer
        self.i2c_bus = smbus.SMBus(1)
        self.motor_feedback_enabled = False
        
        # Initialize granular interfaces only if available
        self._init_motor_current_sensing()
        self._init_individual_motor_control()
    
    def _init_motor_current_sensing(self):
        """Set up motor current sensing via I2C."""
        try:
            # Try to access Robot HAT for motor current
            # (Address 0x14 discovered by hardware_discovery.py)
            test_read = self.i2c_bus.read_byte(0x14)
            self.motor_feedback_enabled = True
            print("✅ Motor current sensing available")
        except:
            print("⚠️  Motor current sensing not available")
    
    def get_motor_current(self):
        """Read actual motor current for proprioceptive feedback."""
        if not self.motor_feedback_enabled:
            return {'left': 0.0, 'right': 0.0}
        
        try:
            # Read motor current from Robot HAT
            # (Exact I2C protocol needs reverse engineering)
            left_current = self.i2c_bus.read_byte_data(0x14, 0x10)  # Example register
            right_current = self.i2c_bus.read_byte_data(0x14, 0x11) # Example register
            
            return {
                'left': left_current * 0.01,   # Convert to amps
                'right': right_current * 0.01
            }
        except:
            return {'left': 0.0, 'right': 0.0}
    
    def differential_control(self, left_speed, right_speed):
        """Independent wheel control for learning experiments."""
        
        # Safety check - fall back to SunFounder if needed
        if abs(left_speed - right_speed) > 30:  # Prevent extreme differential
            print("⚠️  Extreme differential detected, using safe steering")
            avg_speed = (left_speed + right_speed) / 2
            steer_angle = (right_speed - left_speed) * 0.5  # Simple mapping
            self.safety.safe_forward(avg_speed)
            self.safety.safe_steering(steer_angle)
            return
        
        # Use granular control for learning
        try:
            # This would need specific I2C commands to Robot HAT
            # or GPIO PWM to individual motor controllers
            self._set_left_motor(left_speed)
            self._set_right_motor(right_speed)
        except Exception as e:
            # Fall back to safety layer
            print(f"Granular control failed: {e}, using safety layer")
            avg_speed = (left_speed + right_speed) / 2
            self.safety.safe_forward(avg_speed)
```

### Layer 3: Brain Interface (Research Layer)
```python
class BrainMotorInterface:
    """Clean interface for brain server - hides complexity."""
    
    def __init__(self, enhanced_control):
        self.control = enhanced_control
        self.learning_mode = False
    
    def set_learning_mode(self, enabled):
        """Enable/disable granular motor learning."""
        self.learning_mode = enabled
        if enabled:
            print("🧠 Learning mode: Using granular motor control")
        else:
            print("🛡️ Safe mode: Using SunFounder API only")
    
    def move_robot(self, left_speed, right_speed):
        """Brain controls robot movement."""
        
        if self.learning_mode:
            # Use enhanced control for motor learning
            self.control.differential_control(left_speed, right_speed)
            
            # Provide rich feedback for learning
            return {
                'motor_current': self.control.get_motor_current(),
                'command_executed': 'differential',
                'safety_override': False
            }
        else:
            # Use safe SunFounder API
            avg_speed = (left_speed + right_speed) / 2
            steer_angle = (right_speed - left_speed) * 0.5
            
            self.control.safety.safe_forward(avg_speed)
            self.control.safety.safe_steering(steer_angle)
            
            return {
                'motor_current': {'left': 0.0, 'right': 0.0},
                'command_executed': 'sunfounder_api',
                'safety_override': False
            }
```

## 🔬 Research Applications

### Motor Babbling with Feedback
```python
class MotorLearningExperiment:
    """Research experiments using hybrid control."""
    
    def __init__(self, brain_interface):
        self.brain = brain_interface
        self.experiment_data = []
    
    def motor_babbling_experiment(self):
        """Let brain discover motor coordination through feedback."""
        
        self.brain.set_learning_mode(True)
        
        for trial in range(100):
            # Brain generates random motor commands
            left_speed = random.uniform(0, 30)
            right_speed = random.uniform(0, 30)
            
            # Execute command and measure response
            feedback = self.brain.move_robot(left_speed, right_speed)
            
            # Rich proprioceptive feedback
            ultrasonic_before = px.ultrasonic.read()
            time.sleep(0.5)  # Let motion happen
            ultrasonic_after = px.ultrasonic.read()
            
            experiment_data = {
                'trial': trial,
                'command': {'left': left_speed, 'right': right_speed},
                'motor_current': feedback['motor_current'],
                'distance_change': ultrasonic_before - ultrasonic_after,
                'timestamp': time.time()
            }
            
            self.experiment_data.append(experiment_data)
            
            # Brain learns from this rich feedback
            # "When I set left=20, right=30, I moved forward and right, 
            #  left motor drew 0.8A, right motor drew 1.1A"
```

### Coordination Discovery
```python
def discover_straight_line_motion():
    """Brain discovers how to move straight through experimentation."""
    
    brain_interface.set_learning_mode(True)
    
    best_straightness = float('inf')
    best_command = None
    
    for left_power in range(0, 31, 5):
        for right_power in range(0, 31, 5):
            # Test this power combination
            start_position = get_robot_position()
            brain_interface.move_robot(left_power, right_power)
            time.sleep(1.0)
            end_position = get_robot_position()
            
            # Measure how straight the motion was
            straightness = calculate_trajectory_straightness(start_position, end_position)
            
            if straightness < best_straightness:
                best_straightness = straightness
                best_command = (left_power, right_power)
                print(f"🎯 New best straight motion: left={left_power}, right={right_power}")
    
    print(f"🏆 Brain discovered straight motion: {best_command}")
    # Maybe left=25, right=22 due to motor differences!
```

## 🎯 Practical Implementation Strategy

### Phase 1: Hardware Discovery
```bash
# Run on Pi Zero to see what's available
python3 hardware_discovery.py --detailed
```

### Phase 2: SunFounder Foundation
```python
# Start with pure SunFounder API
px = Picarx()
px.forward(30)  # Proven to work
```

### Phase 3: Selective Enhancement
```python
# Add granular control only where meaningful
if motor_current_sensing_available():
    enable_motor_feedback()

if individual_motor_control_available():
    enable_differential_learning()
```

### Phase 4: Research Experiments
```python
# Use hybrid system for motor coordination research
brain.learn_motor_coordination(
    use_feedback=True,
    safety_layer=sunfounder_api,
    granular_layer=i2c_gpio_control
)
```

## 🛡️ Safety Strategy

1. **SunFounder API always available** - Emergency stops always work
2. **Gradual enhancement** - Add granular control incrementally  
3. **Fallback mechanisms** - Revert to safe API if granular fails
4. **Conservative limits** - Cap speeds and angles even in learning mode
5. **Monitoring** - Watch for unsafe behaviors and intervene

## 🎯 Expected Benefits

**Immediate:**
- Working robot using proven SunFounder API
- Safe, reliable basic behaviors

**Research Value:**
- Motor current feedback enables proprioceptive learning
- Individual wheel control enables coordination discovery
- Rich sensor feedback makes learning meaningful

**Future Evolution:**
- Proven foundation for adding more granular interfaces
- Clear path from development to deployment
- Research platform that's also practical

This approach gives you **real motor learning research** while keeping the system practical and safe! 🚀