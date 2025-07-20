# Brain ↔ Brainstem Interface Specification

This document defines the TCP-based communication protocol between the generic brain (cutting-edge machine) and platform-specific brainstems (robot hardware).

## Architecture Overview

```
🧠 BRAIN (cutting edge machine)
   │ Generic field intelligence
   │ Platform-agnostic processing
   │ Abstract stream interface
   │
   ↕ TCP Socket Communication
   │
🌱 BRAINSTEM (robot hardware)  
   │ Platform-specific adaptations
   │ Sensor → stream mapping
   │ Stream → actuator translation
   │ Hardware optimizations
```

## Design Principles

1. **Brain side**: Generic, no platform assumptions, event-driven
2. **Brainstem side**: Platform-specific, all hardware details
3. **Communication**: Asynchronous, fault-tolerant, capability-negotiated
4. **Timing**: Brain operates "when data arrives", not fixed cycles

## Capability Negotiation Protocol

### 1. Connection Handshake

```json
// Brainstem → Brain: Capability Advertisement
{
  "message_type": "capability_advertisement",
  "timestamp": 1643723400.123,
  "brainstem_info": {
    "platform": "PiCar-X",
    "version": "1.2.3",
    "brainstem_id": "picar_001"
  },
  "input_stream": {
    "dimensions": 24,
    "labels": [
      "position_x", "position_y", "position_z",
      "distance_forward", "distance_left", "distance_right", 
      "camera_red", "camera_green", "camera_blue",
      "audio_level", "temperature", "battery_level",
      "encoder_left", "encoder_right", "compass", 
      "gyro_x", "gyro_y", "gyro_z",
      "accel_x", "accel_y", "accel_z",
      "touch", "light_sensor", "proximity"
    ],
    "value_ranges": [
      [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],  // positions
      [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],     // distances
      [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],     // RGB
      [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],     // audio, temp, battery
      [-1.0, 1.0], [-1.0, 1.0], [0.0, 360.0], // motion sensors
      [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],  // gyro
      [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],  // accel
      [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]      // touch, light, proximity
    ],
    "update_frequency_hz": 20.0,
    "reliability": "best_effort"
  },
  "output_stream": {
    "dimensions": 4,
    "labels": [
      "motor_forward", "motor_turn", "motor_tilt", "servo_grip"
    ],
    "value_ranges": [
      [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]
    ],
    "latency_tolerance_ms": 100,
    "command_rate_hz": 50.0
  },
  "capabilities": {
    "persistent_state": true,
    "emergency_stop": true,
    "diagnostic_info": true,
    "configuration_updates": false
  }
}
```

```json
// Brain → Brainstem: Capability Acceptance
{
  "message_type": "capability_acceptance",
  "timestamp": 1643723400.150,
  "accepted": true,
  "brain_info": {
    "version": "2.1.0",
    "field_brain_type": "generic_unified",
    "brain_id": "field_brain_001"
  },
  "stream_config": {
    "input_buffer_size": 100,
    "output_smoothing": true,
    "missing_data_policy": "interpolate",
    "stream_compression": false
  },
  "operational_mode": {
    "processing_mode": "event_driven",
    "field_persistence": true,
    "learning_enabled": true
  }
}
```

### 2. Alternative: Minimal Capability Negotiation

For simpler brainstems that can't provide detailed metadata:

```json
// Minimal brainstem advertisement
{
  "message_type": "simple_capability",
  "timestamp": 1643723400.123,
  "input_dimensions": 12,
  "output_dimensions": 6,
  "brainstem_id": "simple_drone_001"
}
```

## Runtime Communication Protocol

### 3. Sensor Data Streaming

```json
// Brainstem → Brain: Sensor Stream
{
  "message_type": "sensor_stream",
  "timestamp": 1643723400.200,
  "sequence_id": 1001,
  "stream_data": [
    0.25, -0.1, 0.0,        // position x,y,z
    0.8, 0.3, 0.3,          // distances
    0.7, 0.2, 0.1,          // camera RGB
    0.4, 22.5, 0.85,        // audio, temp, battery
    0.1, -0.05, 178.5,      // encoders, compass
    0.02, -0.01, 0.001,     // gyro
    0.05, 0.02, 9.81,       // accelerometer
    0.0, 0.6, 0.2           // touch, light, proximity
  ],
  "data_quality": {
    "missing_sensors": [],
    "sensor_confidence": [1.0, 1.0, 1.0, /* ... 24 values */],
    "timestamp_jitter_ms": 2.3
  }
}
```

### 4. Action Commands

```json
// Brain → Brainstem: Action Stream
{
  "message_type": "action_stream", 
  "timestamp": 1643723400.205,
  "sequence_id": 1001,
  "stream_data": [
    0.6,    // motor_forward: 60% forward
    -0.2,   // motor_turn: 20% left turn
    0.0,    // motor_tilt: no tilt
    0.8     // servo_grip: 80% grip strength
  ],
  "action_metadata": {
    "confidence": 0.85,
    "urgency": "normal",
    "expected_duration_ms": 50,
    "safety_override": false
  }
}
```

### 5. System Messages

```json
// Brain state updates
{
  "message_type": "brain_state",
  "timestamp": 1643723400.300,
  "state": {
    "field_energy": 1247.83,
    "processing_cycles": 15234,
    "topology_regions": 12,
    "learning_rate": 0.15,
    "attention_focus": [0.2, 0.8, 0.1]
  }
}

// Brainstem diagnostics  
{
  "message_type": "brainstem_diagnostics",
  "timestamp": 1643723400.310,
  "diagnostics": {
    "cpu_usage": 0.45,
    "memory_usage": 0.32,
    "sensor_failures": [],
    "actuator_status": "nominal",
    "network_latency_ms": 15.2
  }
}

// Emergency stop
{
  "message_type": "emergency_stop",
  "timestamp": 1643723400.999,
  "reason": "obstacle_detected",
  "stop_all_actuators": true
}
```

## Message Format Standards

### Base Message Structure
```json
{
  "message_type": "string",         // Required: message type identifier
  "timestamp": 1643723400.123,      // Required: Unix timestamp with ms precision
  "sequence_id": 1001,              // Optional: for ordering/deduplication
  // Message-specific payload
}
```

### Stream Data Format
- **Numeric arrays**: Always use arrays of floats, even for single values
- **Value ranges**: Must match negotiated ranges from capability exchange
- **Missing data**: Use `null` for unavailable sensor readings
- **Timestamps**: Unix time with millisecond precision

### Error Handling
```json
{
  "message_type": "error",
  "timestamp": 1643723400.400,
  "error_code": "STREAM_TIMEOUT",
  "error_message": "No sensor data received for 500ms",
  "recovery_action": "using_interpolated_data"
}
```

## Implementation Guidelines

### Brain Side (Generic)
1. **No platform assumptions**: Work with any input/output dimensions
2. **Event-driven**: Process when data arrives, don't assume timing
3. **Fault tolerant**: Handle missing data, network issues gracefully
4. **Capability adaptive**: Adapt field mappings to any stream configuration
5. **Asynchronous**: Non-blocking I/O, buffered streams

### Brainstem Side (Platform-Specific)
1. **Sensor abstraction**: Map hardware sensors to normalized streams
2. **Actuator translation**: Convert stream commands to hardware actions
3. **Hardware optimization**: Platform-specific performance optimizations
4. **Safety systems**: Emergency stops, boundary checking, failsafes
5. **Diagnostic monitoring**: Hardware health, performance metrics

## Network Considerations

### TCP Socket Configuration
- **Port**: Configurable, default 8889
- **Keep-alive**: Enabled with 30s timeout
- **Buffer size**: 64KB for low-latency streaming
- **No-delay**: Enabled (TCP_NODELAY) for real-time response

### Fault Tolerance
- **Connection drops**: Auto-reconnect with exponential backoff
- **Message loss**: Sequence IDs for detection, not guaranteed delivery
- **Latency spikes**: Graceful degradation, temporal interpolation
- **Brain crashes**: Brainstem continues safe operation independently

### Security
- **Authentication**: Optional shared secret for connection
- **Encryption**: TLS optional for sensitive applications
- **Network isolation**: Recommend dedicated network segment

## Example Platform Implementations

### PiCar-X Brainstem
```python
# Hardware-specific sensor mapping
sensors = {
    'ultrasonic': map_to_distance_sensors(),
    'camera': map_to_rgb_stream(),
    'imu': map_to_motion_sensors(),
    'motors': map_from_action_stream()
}
```

### Drone Brainstem
```python
# Different platform, different mapping
sensors = {
    'gps': map_to_position_stream(),
    'imu': map_to_orientation_stream(),
    'rotors': map_from_thrust_commands()
}
```

### Simulation Brainstem
```python
# Virtual platform for testing
sensors = {
    'physics_engine': map_to_simulated_sensors(),
    'virtual_actuators': map_from_simulated_actions()
}
```

## Migration Path

1. **Current robot-specific brain**: Can work with compatibility brainstem
2. **Compatibility brainstem**: Translates old protocol to new interface
3. **New generic brain**: Works with any compliant brainstem
4. **Future platforms**: Only need brainstem implementation, brain is universal

This interface enables the same brain to control any platform - ground robots, drones, manipulators, or simulated environments - with only brainstem-side changes.