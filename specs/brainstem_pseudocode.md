# Brainstem (Hardware Interface) Pseudocode

## Overview
The brainstem runs on the Raspberry Pi and handles all direct hardware interaction. It maintains real-time responsiveness while the brain deliberates on the laptop. Key responsibilities include sensor reading, motor control, client-side prediction, and basic safety.

## Main Brainstem Loop
```python
def main_brainstem_loop(hardware_config, brain_connection):
    """
    Real-time loop handling immediate hardware needs
    Runs at high frequency (50-100Hz) for responsive control
    """
    # Hardware discovery and initialization
    sensors = discover_and_initialize_sensors()
    actuators = discover_and_initialize_actuators()
    
    # Send hardware manifest to brain
    send_hardware_capabilities(brain_connection, sensors, actuators)
    
    # Initialize state
    current_motor_commands = get_safe_default_commands(actuators)
    last_brain_command_time = current_time()
    prediction_engine = ClientSidePrediction()
    
    while robot_is_active():
        cycle_start_time = current_time()
        
        # === SENSOR READING ===
        sensor_readings = read_all_sensors(sensors)
        actuator_positions = read_actuator_positions(actuators)
        
        # === COMMUNICATION WITH BRAIN ===
        # Check for new commands from brain
        new_brain_command = check_for_brain_commands(brain_connection, non_blocking=True)
        
        if new_brain_command:
            current_motor_commands = new_brain_command['actuator_commands']
            last_brain_command_time = current_time()
            prediction_engine.update_brain_command(new_brain_command)
        
        # Send current sensor state to brain
        sensory_packet = create_sensory_packet(
            sensor_readings, 
            actuator_positions,
            cycle_start_time
        )
        send_to_brain(brain_connection, sensory_packet, non_blocking=True)
        
        # === CLIENT-SIDE PREDICTION ===
        # If brain is taking too long, predict what it would want
        time_since_brain_command = current_time() - last_brain_command_time
        
        if time_since_brain_command > brain_timeout_threshold():
            predicted_commands = prediction_engine.predict_next_commands(
                sensor_readings, 
                current_motor_commands
            )
            current_motor_commands = predicted_commands
        
        # === SAFETY CHECKS ===
        safe_commands = apply_safety_constraints(
            current_motor_commands, 
            sensor_readings,
            actuators
        )
        
        # === MOTOR EXECUTION ===
        execute_motor_commands(actuators, safe_commands)
        
        # === TIMING ===
        maintain_cycle_frequency(cycle_start_time, target_frequency=50)  # 50Hz
```

## Hardware Discovery
```python
def discover_and_initialize_sensors():
    """
    Enumerate all available sensors and their capabilities
    Returns standardized sensor interface
    """
    sensors = []
    
    # Camera detection
    for camera_id in range(4):  # Check first 4 camera slots
        try:
            camera = initialize_camera(camera_id)
            if camera.is_functional():
                sensors.append({
                    'type': 'camera',
                    'id': f'camera_{camera_id}',
                    'data_type': 'image_features',  # Processed, not raw pixels
                    'update_frequency': 30,  # Hz
                    'interface': camera
                })
        except:
            continue
    
    # Ultrasonic sensors
    ultrasonic_pins = [(18, 24), (23, 25)]  # (trigger, echo) pairs
    for i, (trigger_pin, echo_pin) in enumerate(ultrasonic_pins):
        try:
            ultrasonic = initialize_ultrasonic(trigger_pin, echo_pin)
            sensors.append({
                'type': 'distance',
                'id': f'ultrasonic_{i}',
                'data_type': 'float',
                'range': (0.02, 4.0),  # meters
                'update_frequency': 20,
                'interface': ultrasonic
            })
        except:
            continue
    
    # Add any other sensors found by the system
    # IMU, light sensors, microphones, etc.
    
    return sensors

def discover_and_initialize_actuators():
    """
    Enumerate all available actuators and their safe operating ranges
    """
    actuators = []
    
    # Servo motors (steering, camera pan/tilt)
    servo_pins = [2, 3, 4]  # PWM pins
    for i, pin in enumerate(servo_pins):
        try:
            servo = initialize_servo(pin)
            actuators.append({
                'type': 'servo',
                'id': f'servo_{i}',
                'data_type': 'float',
                'range': (-1.0, 1.0),  # Normalized -1 to +1
                'safe_default': 0.0,
                'max_change_per_step': 0.1,
                'interface': servo
            })
        except:
            continue
    
    # Drive motors (left, right wheels)
    motor_pins = [(5, 6), (7, 8)]  # (forward, backward) pairs
    for i, (fwd_pin, bwd_pin) in enumerate(motor_pins):
        try:
            motor = initialize_motor(fwd_pin, bwd_pin)
            actuators.append({
                'type': 'motor',
                'id': f'motor_{i}',
                'data_type': 'float',
                'range': (-1.0, 1.0),  # -1 = full reverse, +1 = full forward
                'safe_default': 0.0,
                'max_change_per_step': 0.2,
                'interface': motor
            })
        except:
            continue
    
    return actuators
```

## Client-Side Prediction
```python
class ClientSidePrediction:
    """
    Predict what the brain would command during network delays
    Simple extrapolation to maintain smooth control
    """
    
    def __init__(self):
        self.command_history = []
        self.sensor_history = []
        self.prediction_accuracy = 0.5  # Track how well we predict
        
    def update_brain_command(self, brain_command):
        """
        Record actual brain command to improve prediction accuracy
        """
        if self.command_history:
            # Check how accurate our last prediction was
            last_prediction = self.command_history[-1].get('predicted', {})
            actual_command = brain_command['actuator_commands']
            
            accuracy = self.calculate_prediction_accuracy(last_prediction, actual_command)
            self.prediction_accuracy = (self.prediction_accuracy * 0.9 + accuracy * 0.1)
        
        # Store this command
        self.command_history.append({
            'commands': brain_command['actuator_commands'],
            'timestamp': current_time(),
            'predicted': False
        })
        
        # Keep only recent history
        if len(self.command_history) > 10:
            self.command_history = self.command_history[-5:]
    
    def predict_next_commands(self, current_sensors, current_commands):
        """
        Predict what the brain would command given current sensor state
        """
        if len(self.command_history) < 2:
            # Not enough history - just continue current commands
            return current_commands
        
        # Simple approach: extrapolate recent command trends
        recent_commands = [entry['commands'] for entry in self.command_history[-3:]]
        
        predicted_commands = {}
        for actuator_id in current_commands:
            # Get recent values for this actuator
            recent_values = [cmd.get(actuator_id, 0.0) for cmd in recent_commands]
            
            if len(recent_values) >= 2:
                # Linear extrapolation
                trend = recent_values[-1] - recent_values[-2]
                predicted_value = recent_values[-1] + (trend * 0.5)  # Conservative
                
                # Clamp to safe range
                predicted_commands[actuator_id] = max(-1.0, min(1.0, predicted_value))
            else:
                predicted_commands[actuator_id] = current_commands[actuator_id]
        
        # Mark as prediction for accuracy tracking
        self.command_history.append({
            'commands': predicted_commands,
            'timestamp': current_time(),
            'predicted': True
        })
        
        return predicted_commands
    
    def calculate_prediction_accuracy(self, predicted, actual):
        """
        Calculate how close our prediction was to actual brain command
        """
        if not predicted or not actual:
            return 0.0
        
        total_error = 0.0
        count = 0
        
        for actuator_id in actual:
            if actuator_id in predicted:
                error = abs(predicted[actuator_id] - actual[actuator_id])
                total_error += error
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_error = total_error / count
        accuracy = max(0.0, 1.0 - avg_error)  # Convert error to accuracy
        return accuracy
```

## Safety Systems
```python
def apply_safety_constraints(motor_commands, sensor_readings, actuators):
    """
    Apply safety limits to prevent hardware damage or dangerous behavior
    """
    safe_commands = motor_commands.copy()
    
    # === RATE LIMITING ===
    # Prevent sudden large changes that could damage hardware
    for actuator in actuators:
        actuator_id = actuator['id']
        if actuator_id not in safe_commands:
            continue
            
        max_change = actuator['max_change_per_step']
        current_position = actuator['interface'].get_current_position()
        desired_position = safe_commands[actuator_id]
        
        change = desired_position - current_position
        if abs(change) > max_change:
            # Limit the change rate
            limited_change = max_change if change > 0 else -max_change
            safe_commands[actuator_id] = current_position + limited_change
    
    # === RANGE CLAMPING ===
    # Ensure commands are within safe operating ranges
    for actuator in actuators:
        actuator_id = actuator['id']
        if actuator_id not in safe_commands:
            continue
            
        min_val, max_val = actuator['range']
        safe_commands[actuator_id] = max(min_val, min(max_val, safe_commands[actuator_id]))
    
    # === COLLISION AVOIDANCE ===
    # Emergency stop if obstacle too close
    for sensor_reading in sensor_readings:
        if sensor_reading.get('type') == 'distance':
            distance = sensor_reading['value']
            if distance < 0.05:  # 5cm emergency threshold
                # Stop all motors immediately
                for actuator in actuators:
                    if actuator['type'] == 'motor':
                        safe_commands[actuator['id']] = 0.0
                break
    
    # === POWER MANAGEMENT ===
    # Reduce power if battery low (if available)
    battery_level = get_battery_level()
    if battery_level < 0.2:  # Below 20%
        # Reduce all motor commands to conserve power
        for actuator in actuators:
            if actuator['type'] == 'motor':
                actuator_id = actuator['id']
                if actuator_id in safe_commands:
                    safe_commands[actuator_id] *= 0.5
    
    return safe_commands

def get_safe_default_commands(actuators):
    """
    Generate safe default commands for all actuators
    """
    default_commands = {}
    
    for actuator in actuators:
        default_commands[actuator['id']] = actuator['safe_default']
    
    return default_commands

def brain_timeout_threshold():
    """
    How long to wait before using client-side prediction
    """
    return 0.2  # 200ms - if brain takes longer, predict locally
```

## Sensor Processing
```python
def read_all_sensors(sensors):
    """
    Read current values from all sensors
    """
    sensor_readings = []
    
    for sensor in sensors:
        try:
            if sensor['type'] == 'camera':
                # Process camera into feature vector, not raw pixels
                image_features = process_camera_features(sensor['interface'])
                sensor_readings.append({
                    'sensor_id': sensor['id'],
                    'type': 'camera_features',
                    'values': image_features,  # List of floats
                    'timestamp': current_time()
                })
                
            elif sensor['type'] == 'distance':
                distance = sensor['interface'].read_distance()
                sensor_readings.append({
                    'sensor_id': sensor['id'],
                    'type': 'distance',
                    'values': [distance],  # Single float in list for consistency
                    'timestamp': current_time()
                })
                
            # Add other sensor types as needed
            
        except Exception as e:
            # Sensor failed - report error value
            sensor_readings.append({
                'sensor_id': sensor['id'],
                'type': 'error',
                'values': [-999.0],  # Error sentinel value
                'timestamp': current_time(),
                'error': str(e)
            })
    
    return sensor_readings

def process_camera_features(camera):
    """
    Convert camera image to feature vector for brain processing
    Reduces bandwidth and processing load
    """
    # Capture image
    image = camera.capture_frame()
    
    if image is None:
        return [0.0] * 32  # Return zero features if camera fails
    
    # Simple feature extraction - could be made more sophisticated
    features = []
    
    # Basic color histograms (RGB)
    for channel in range(3):  # R, G, B
        hist = calculate_histogram(image[:,:,channel], bins=8)
        features.extend(hist)  # 8 values per channel = 24 total
    
    # Simple edge detection response
    edges = detect_edges(image)
    edge_strength = [
        edges[:image.shape[0]//2, :].mean(),      # Top half
        edges[image.shape[0]//2:, :].mean(),      # Bottom half
        edges[:, :image.shape[1]//2].mean(),      # Left half
        edges[:, image.shape[1]//2:].mean(),      # Right half
    ]
    features.extend(edge_strength)  # 4 values
    
    # Motion detection (compare with previous frame)
    motion = detect_motion(image, camera.get_previous_frame())
    features.extend([motion.mean(), motion.max()])  # 2 values
    
    # Brightness and contrast
    brightness = image.mean()
    contrast = image.std()
    features.extend([brightness / 255.0, contrast / 255.0])  # 2 values, normalized
    
    # Total: 24 + 4 + 2 + 2 = 32 features
    return features

def read_actuator_positions(actuators):
    """
    Get current positions of all actuators for feedback
    """
    positions = []
    
    for actuator in actuators:
        try:
            position = actuator['interface'].get_current_position()
            positions.append({
                'actuator_id': actuator['id'],
                'position': position,
                'timestamp': current_time()
            })
        except:
            positions.append({
                'actuator_id': actuator['id'],
                'position': -999.0,  # Error sentinel
                'timestamp': current_time()
            })
    
    return positions
```

## Communication Protocol
```python
def create_sensory_packet(sensor_readings, actuator_positions, timestamp):
    """
    Package sensor data for transmission to brain
    """
    # Flatten all sensor values into single list
    all_sensor_values = []
    
    for reading in sensor_readings:
        all_sensor_values.extend(reading['values'])
    
    # Flatten actuator positions
    all_actuator_positions = [pos['position'] for pos in actuator_positions]
    
    return {
        'sensor_values': all_sensor_values,
        'actuator_positions': all_actuator_positions,
        'timestamp': timestamp,
        'sequence_id': generate_sequence_id(),
        'brainstem_id': get_brainstem_id()
    }

def send_to_brain(connection, packet, non_blocking=True):
    """
    Send data to brain over network
    """
    try:
        if non_blocking:
            connection.send_async(packet)
        else:
            connection.send(packet)
    except NetworkError:
        # Log error but continue - brain will handle missing data
        log_network_error("Failed to send to brain")

def check_for_brain_commands(connection, non_blocking=True):
    """
    Check for new commands from brain
    """
    try:
        if non_blocking:
            return connection.receive_async()  # Returns None if no data
        else:
            return connection.receive()
    except NetworkError:
        return None

def send_hardware_capabilities(connection, sensors, actuators):
    """
    Send hardware manifest to brain on startup
    """
    capability_packet = {
        'message_type': 'hardware_capabilities',
        'sensors': [
            {
                'id': sensor['id'],
                'type': sensor['type'],
                'data_size': len(sensor.get('sample_data', [0.0])),
                'update_frequency': sensor['update_frequency']
            }
            for sensor in sensors
        ],
        'actuators': [
            {
                'id': actuator['id'],
                'type': actuator['type'],
                'range': actuator['range'],
                'safe_default': actuator['safe_default']
            }
            for actuator in actuators
        ],
        'brainstem_id': get_brainstem_id(),
        'timestamp': current_time()
    }
    
    send_to_brain(connection, capability_packet, non_blocking=False)
```

## Utility Functions
```python
def execute_motor_commands(actuators, commands):
    """
    Apply motor commands to physical hardware
    """
    for actuator in actuators:
        actuator_id = actuator['id']
        if actuator_id in commands:
            try:
                command_value = commands[actuator_id]
                actuator['interface'].set_position(command_value)
            except Exception as e:
                log_actuator_error(f"Failed to control {actuator_id}: {e}")

def maintain_cycle_frequency(cycle_start_time, target_frequency):
    """
    Sleep to maintain consistent loop timing
    """
    cycle_duration = current_time() - cycle_start_time
    target_duration = 1.0 / target_frequency
    
    if cycle_duration < target_duration:
        sleep_time = target_duration - cycle_duration
        time.sleep(sleep_time)
    elif cycle_duration > target_duration * 1.5:
        log_performance_warning(f"Brainstem cycle overrun: {cycle_duration:.3f}s")

def get_battery_level():
    """
    Check battery level if available
    """
    try:
        # Implementation depends on hardware
        return read_battery_voltage() / max_battery_voltage()
    except:
        return 1.0  # Assume full if can't read

def generate_sequence_id():
    """
    Generate monotonic sequence ID for message ordering
    """
    if not hasattr(generate_sequence_id, 'counter'):
        generate_sequence_id.counter = 0
    generate_sequence_id.counter += 1
    return generate_sequence_id.counter

def get_brainstem_id():
    """
    Unique identifier for this brainstem instance
    """
    # Could use MAC address, serial number, etc.
    return "brainstem_001"

# Hardware abstraction functions (platform specific)
def initialize_camera(camera_id):
    """Platform-specific camera initialization"""
    pass

def initialize_ultrasonic(trigger_pin, echo_pin):
    """Platform-specific ultrasonic sensor initialization"""
    pass

def initialize_servo(pin):
    """Platform-specific servo initialization"""
    pass

def initialize_motor(forward_pin, backward_pin):
    """Platform-specific motor initialization"""
    pass
```