# Brain-Brainstem Communication Protocol

## Overview
Simple, efficient protocol for real-time communication between the brain (laptop) and brainstem (Raspberry Pi). Designed for low latency and graceful degradation under network issues.

## Message Types

### 1. Hardware Capabilities (Brainstem → Brain)
Sent once at startup to inform brain about available hardware.

```python
{
    "message_type": "hardware_capabilities",
    "brainstem_id": "brainstem_001",
    "timestamp": 1642234567.123,
    "sensors": [
        {
            "id": "camera_0",
            "type": "camera_features", 
            "data_size": 32,           # Number of float values
            "update_frequency": 30     # Hz
        },
        {
            "id": "ultrasonic_0",
            "type": "distance",
            "data_size": 1,
            "update_frequency": 20
        },
        {
            "id": "ultrasonic_1", 
            "type": "distance",
            "data_size": 1,
            "update_frequency": 20
        }
    ],
    "actuators": [
        {
            "id": "servo_0",           # Steering
            "type": "servo",
            "range": [-1.0, 1.0],
            "safe_default": 0.0
        },
        {
            "id": "motor_0",           # Left wheel
            "type": "motor", 
            "range": [-1.0, 1.0],
            "safe_default": 0.0
        },
        {
            "id": "motor_1",           # Right wheel
            "type": "motor",
            "range": [-1.0, 1.0], 
            "safe_default": 0.0
        }
    ]
}
```

### 2. Sensory Data (Brainstem → Brain)
Continuous stream of sensor readings sent at high frequency (~50Hz).

```python
{
    "message_type": "sensory_data",
    "brainstem_id": "brainstem_001", 
    "sequence_id": 12345,
    "timestamp": 1642234567.456,
    
    # Flattened sensor values in consistent order
    "sensor_values": [
        # Camera features (32 values)
        0.23, 0.45, 0.12, ...,
        # Ultrasonic 0 (1 value) 
        1.47,
        # Ultrasonic 1 (1 value)
        0.89
    ],
    
    # Current actuator positions for feedback
    "actuator_positions": [
        0.15,    # servo_0 current position
        0.0,     # motor_0 current position  
        0.0      # motor_1 current position
    ],
    
    # Optional: brainstem's prediction of what brain wants
    "brainstem_prediction": {
        "servo_0": 0.12,
        "motor_0": 0.0,
        "motor_1": 0.0
    },
    
    # Network performance info
    "network_latency": 0.023  # Measured round-trip time
}
```

### 3. Motor Commands (Brain → Brainstem)
Commands for actuator control sent as predictions are generated.

```python
{
    "message_type": "motor_commands",
    "brain_id": "brain_001",
    "sequence_id": 12346,
    "timestamp": 1642234567.789,
    
    # Direct actuator commands
    "actuator_commands": {
        "servo_0": 0.25,     # Steering angle
        "motor_0": 0.3,      # Left wheel speed
        "motor_1": 0.3       # Right wheel speed
    },
    
    # Optional: Brain's confidence in these commands
    "confidence": 0.85,
    
    # Optional: Expected duration of these commands
    "duration_estimate": 0.1  # Seconds
}
```

### 4. Status Messages (Bidirectional)
Health monitoring and error reporting.

```python
{
    "message_type": "status",
    "sender_id": "brainstem_001", 
    "timestamp": 1642234567.999,
    
    "status": "healthy",  # "healthy", "warning", "error"
    
    # Performance metrics
    "metrics": {
        "cpu_usage": 0.45,
        "memory_usage": 0.32,
        "loop_frequency": 49.8,  # Actual Hz
        "battery_level": 0.78     # If available
    },
    
    # Any errors or warnings
    "messages": [
        "Camera_0 intermittent failures",
        "Network latency high: 150ms"
    ]
}
```

## Protocol Characteristics

### Network Transport
- **UDP for sensor data** - Low latency, can tolerate occasional packet loss
- **TCP for commands and status** - Reliable delivery for critical messages
- **JSON encoding** - Human readable, easy to debug
- **Message compression** - Gzip for larger payloads if needed

### Timing and Synchronization
- **Brainstem frequency**: 50Hz (20ms cycles)
- **Brain frequency**: Variable based on thinking time
- **Sequence IDs**: Monotonic counters for message ordering
- **Timestamps**: High precision for latency measurement

### Error Handling
- **Missing brain commands**: Brainstem continues last command + client-side prediction
- **Network disconnection**: Brainstem enters safe mode (stop all motors)
- **Malformed messages**: Log error, ignore message, continue operation
- **Sequence gaps**: Detect lost messages but continue (UDP tolerance)

## Client-Side Prediction Protocol

When the brain is slow to respond, the brainstem predicts commands:

```python
# Brainstem logic for prediction
def handle_missing_brain_command():
    time_since_last_command = current_time() - last_brain_command_time
    
    if time_since_last_command < 0.1:  # 100ms
        # Recent command - continue current action
        return current_motor_commands
        
    elif time_since_last_command < 0.5:  # 500ms
        # Use client-side prediction
        return predict_next_commands()
        
    else:  # > 500ms
        # Brain unresponsive - safe mode
        return safe_stop_commands()
```

## Network Optimization

### Bandwidth Management
- **Sensor data**: ~40 floats × 4 bytes × 50Hz = 8KB/s
- **Motor commands**: ~3 floats × 4 bytes × variable Hz = <1KB/s  
- **Status messages**: Occasional, <1KB each
- **Total bandwidth**: <10KB/s typical

### Latency Optimization
- **Message prioritization**: Motor commands > sensor data > status
- **Adaptive frequency**: Reduce sensor frequency if network congested
- **Local buffering**: Brainstem buffers recent data for smoothing

## Implementation Examples

### Brain Connection Handler
```python
class BrainConnection:
    def __init__(self, brainstem_ip, brain_ip):
        self.sensor_socket = UDPSocket(brain_ip, 5001)  # High frequency sensor data
        self.command_socket = TCPSocket(brain_ip, 5002)  # Reliable commands
        self.status_socket = TCPSocket(brain_ip, 5003)   # Status and health
        
    def send_sensor_data(self, sensor_packet):
        """Send sensor data via UDP (lossy but fast)"""
        self.sensor_socket.send(json.dumps(sensor_packet))
        
    def receive_motor_commands(self):
        """Receive motor commands via TCP (reliable)"""
        try:
            data = self.command_socket.receive(timeout=0.01)  # Non-blocking
            return json.loads(data) if data else None
        except timeout:
            return None
```

### Message Validation
```python
def validate_motor_command(command_msg, actuators):
    """Validate incoming motor command for safety"""
    
    # Check message structure
    required_fields = ['message_type', 'actuator_commands', 'timestamp']
    if not all(field in command_msg for field in required_fields):
        return False, "Missing required fields"
    
    # Check actuator commands
    commands = command_msg['actuator_commands']
    for actuator_id, value in commands.items():
        
        # Verify actuator exists
        actuator = find_actuator_by_id(actuators, actuator_id)
        if not actuator:
            return False, f"Unknown actuator: {actuator_id}"
        
        # Verify value in range
        min_val, max_val = actuator['range']
        if not (min_val <= value <= max_val):
            return False, f"Value {value} out of range for {actuator_id}"
    
    return True, "Valid"
```

## Security Considerations

### Authentication
- **Pre-shared key**: Simple symmetric key for message authentication
- **Source validation**: Only accept commands from known brain IP
- **Message signing**: HMAC signatures for critical motor commands

### Safety Boundaries  
- **Command validation**: All motor commands checked against safe ranges
- **Rate limiting**: Prevent command flooding attacks
- **Emergency override**: Physical emergency stop always available

## Debugging and Monitoring

### Message Logging
```python
# Log all messages for debugging
def log_message(direction, message_type, content):
    timestamp = current_time()
    log_entry = {
        'timestamp': timestamp,
        'direction': direction,  # 'sent' or 'received'
        'type': message_type,
        'content': content,
        'size': len(json.dumps(content))
    }
    
    debug_logger.write(log_entry)
```

### Performance Monitoring
- **Latency tracking**: Measure round-trip times
- **Throughput monitoring**: Track messages/second  
- **Error rates**: Count malformed/dropped messages
- **Prediction accuracy**: How well client-side prediction works

This protocol provides the communication foundation for the brain-brainstem architecture while maintaining simplicity and robustness.