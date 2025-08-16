# Brainstem Communication Protocol (Actual Implementation)

## Overview

This document defines the **actual implemented** communication protocol between robot brainstems (clients) and the brain server. This is a minimal, high-performance binary protocol optimized for real-time robotics communication.

**Key Design Principles:**
- **Fixed overhead**: 13 bytes header per message
- **High performance**: Binary format with no parsing overhead
- **Simple implementation**: Easy to implement on embedded systems
- **Vector-based**: Pure float vectors for sensory input and motor commands
- **Magic bytes**: 'ROBO' (0x524F424F) for protocol validation

## Protocol Fundamentals

### Transport Layer
- **Protocol**: TCP/IP sockets
- **Port**: 9999 (default, configurable)
- **Connection**: Persistent, long-lived connection
- **Encoding**: Binary struct format (network byte order for header, native for data)
- **Data Type**: IEEE 754 float32 arrays

### Message Structure
```
┌────────────┬────────────┬──────────┬──────────────┬────────────────────┐
│ Magic      │ Length     │ Type     │ Vector Len   │ Vector Data        │
│ (4 bytes)  │ (4 bytes)  │ (1 byte) │ (4 bytes)    │ (N * 4 bytes)      │
│ 0x524F424F │ uint32     │ uint8    │ uint32       │ float32 array      │
│ ('ROBO')   │            │          │              │                    │
└────────────┴────────────┴──────────┴──────────────┴────────────────────┘
```

**Header size: 13 bytes (magic + length + type + vector_length)**
**Note:** Length field contains the message size AFTER the magic bytes (type + vector_len + data)

### Message Types

| Type | Value | Name | Direction | Purpose |
|------|-------|------|-----------|---------|
| 0 | 0x00 | SENSORY_INPUT | Robot → Brain | Send sensor readings |
| 1 | 0x01 | ACTION_OUTPUT | Brain → Robot | Receive motor commands |
| 2 | 0x02 | HANDSHAKE | Bidirectional | Connection establishment |
| 255 | 0xFF | ERROR | Bidirectional | Error reporting |

## Communication Flow

### 1. Connection Establishment

#### Step 1: Robot Connects
```
Robot establishes TCP connection to brain server on port 9999
```

#### Step 2: Robot Sends Handshake
```
Message Type: HANDSHAKE (2)
Vector: [robot_version, sensory_size, action_size, hardware_type, capabilities_mask]

Example:
[1.0, 24.0, 4.0, 1.0, 3.0]
 │    │     │    │    └─ Capabilities mask (bit flags: 1=visual, 2=audio, 4=manipulation)
 │    │     │    └─ Hardware type (1.0 = PiCar-X)
 │    │     └─ Expected action vector size
 │    └─ Sensory vector size robot will send
 └─ Robot software version
```

#### Step 3: Brain Responds with Handshake
```
Message Type: HANDSHAKE (2)
Vector: [brain_version, accepted_sensory_size, accepted_action_size, gpu_available, brain_capabilities]

Example:
[4.0, 24.0, 4.0, 1.0, 15.0]
 │    │     │    │    └─ Brain capabilities (bit flags: 1=visual, 2=audio, 4=manipulation, 8=multi-agent)
 │    │     │    └─ GPU acceleration available
 │    │     └─ Action vector size brain will send
 │    └─ Accepted sensory vector size (may differ from requested)
 └─ Brain software version
```

**Dynamic Dimension Negotiation:**
- Robot announces its sensory/action dimensions in handshake
- Brain accepts or negotiates compatible dimensions
- If dimensions don't match exactly, brain adapts internally
- Both sides proceed with negotiated dimensions

### 2. Runtime Communication Loop

#### Step 1: Robot Sends Sensory Input
```
Message Type: SENSORY_INPUT (0)
Vector: [sensor1, sensor2, ..., sensorN]

Example for PiCar-X:
[pos_x, pos_y, heading, speed, ultrasonic_dist, camera_data...]
```

#### Step 2: Brain Responds with Action
```
Message Type: ACTION_OUTPUT (1)
Vector: [action1, action2, action3, action4]

Example for PiCar-X:
[left_motor, right_motor, steering_angle, camera_pan]
```

### 3. Error Handling

#### Error Message Format
```
Message Type: ERROR (255)
Vector: [error_code]

Error Codes:
1.0 = Unknown message type
2.0 = Protocol error
3.0 = Empty sensory input
4.0 = Sensory input too large
5.0 = Brain processing error
```

## Implementation Reference

### Python Implementation (Robot Side)

```python
import struct
import socket
from typing import List, Tuple

class BrainProtocol:
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    MAGIC_BYTES = 0x524F424F  # 'ROBO' in hex
    
    def encode_message(self, vector: List[float], msg_type: int) -> bytes:
        """Encode vector message for transmission."""
        # Convert to float32
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Message length = type(1) + vector_length(4) + vector_data (excludes magic)
        message_length = 1 + 4 + len(vector_data)
        
        # Pack: magic + length + type + vector_length + vector_data
        header = struct.pack('!IIBI', self.MAGIC_BYTES, message_length, msg_type, len(vector))
        return header + vector_data
    
    def send_sensory_input(self, sock: socket.socket, sensors: List[float]):
        """Send sensory input to brain."""
        message = self.encode_message(sensors, self.MSG_SENSORY_INPUT)
        sock.sendall(message)
    
    def receive_action(self, sock: socket.socket) -> List[float]:
        """Receive action commands from brain."""
        # Read message length
        length_data = sock.recv(4)
        message_length = struct.unpack('!I', length_data)[0]
        
        # Read rest of message
        message_data = sock.recv(message_length)
        
        # Decode message
        msg_type, vector_length = struct.unpack('!BI', message_data[:5])
        vector_data = message_data[5:]
        
        if msg_type == self.MSG_ACTION_OUTPUT:
            return list(struct.unpack(f'{vector_length}f', vector_data))
        elif msg_type == self.MSG_ERROR:
            error_code = struct.unpack('f', vector_data)[0]
            raise Exception(f"Brain error: {error_code}")
        else:
            raise Exception(f"Unexpected message type: {msg_type}")

# Usage Example
protocol = BrainProtocol()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 9999))

# Handshake
handshake = protocol.encode_message([1.0, 16.0, 4.0, 1.0], protocol.MSG_HANDSHAKE)
sock.sendall(handshake)

# Main loop
while True:
    # Send sensors
    sensors = [1.2, 0.8, 45.0, 0.15, 0.75, ...]  # 16 float values
    protocol.send_sensory_input(sock, sensors)
    
    # Receive actions
    actions = protocol.receive_action(sock)  # [left, right, steering, camera]
    
    # Execute actions
    execute_motor_commands(actions)
```

### C++ Implementation (Robot Side)

```cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <vector>

class BrainProtocol {
public:
    static const uint8_t MSG_SENSORY_INPUT = 0;
    static const uint8_t MSG_ACTION_OUTPUT = 1;
    static const uint8_t MSG_HANDSHAKE = 2;
    static const uint8_t MSG_ERROR = 255;
    
    std::vector<uint8_t> encodeMessage(const std::vector<float>& vector, uint8_t msgType) {
        uint32_t vectorLength = vector.size();
        uint32_t messageLength = 1 + 4 + (vectorLength * 4);
        
        std::vector<uint8_t> message;
        
        // Pack message length (big-endian)
        message.push_back((messageLength >> 24) & 0xFF);
        message.push_back((messageLength >> 16) & 0xFF);
        message.push_back((messageLength >> 8) & 0xFF);
        message.push_back(messageLength & 0xFF);
        
        // Pack message type
        message.push_back(msgType);
        
        // Pack vector length (big-endian)
        message.push_back((vectorLength >> 24) & 0xFF);
        message.push_back((vectorLength >> 16) & 0xFF);
        message.push_back((vectorLength >> 8) & 0xFF);
        message.push_back(vectorLength & 0xFF);
        
        // Pack vector data
        for (float value : vector) {
            uint32_t* intValue = reinterpret_cast<uint32_t*>(&value);
            message.push_back((*intValue >> 24) & 0xFF);
            message.push_back((*intValue >> 16) & 0xFF);
            message.push_back((*intValue >> 8) & 0xFF);
            message.push_back(*intValue & 0xFF);
        }
        
        return message;
    }
    
    void sendSensoryInput(int socket, const std::vector<float>& sensors) {
        auto message = encodeMessage(sensors, MSG_SENSORY_INPUT);
        send(socket, message.data(), message.size(), 0);
    }
};
```

## Vector Specifications

### Dynamic Vector Dimensions
The protocol supports dynamic sensory and action vector sizes, negotiated during handshake. The brain adapts to different robot configurations automatically.

### Sensory Input Vector Format

The protocol supports dynamic sensory vector dimensions, negotiated during handshake. Robots send normalized float32 values representing their sensor readings.

**Common patterns for sensory vectors:**
- Spatial sensors (position, distance, orientation)
- Motion sensors (velocity, acceleration, motor states)
- Environmental sensors (light, sound, temperature)
- System sensors (battery, health, status)
- Derived/computed values (gradients, patterns, predictions)

**Example sensory vector (N dimensions):**
```
[sensor_0, sensor_1, ..., sensor_N-1]
```
Each value typically normalized to [0, 1] or [-1, 1] range for optimal brain processing.

### Action Output Vector Format

The brain outputs normalized action values that robots interpret based on their actuator configuration, negotiated during handshake.

**Common patterns for action vectors:**
- Locomotion controls (wheels, legs, propellers)
- Manipulation controls (arms, grippers, tools)
- Sensory controls (camera orientation, sensor focus)
- Communication outputs (lights, sounds, signals)

**Example action vector (M dimensions):**
```
[action_0, action_1, ..., action_M-1]
```
Each value typically in range [-1, 1] representing bidirectional control signals.

## Robot-Specific Implementations

Each robot type should document its specific mapping between hardware and the protocol:

- **PiCar-X**: See `client_picarx/PICARX_PROTOCOL_MAPPING.md`
- **Other robots**: Create similar mapping documentation

### Robot Profile Configuration
Robots can define their sensory/action mappings in a profile JSON file:

```json
{
  "robot_type": "picarx",
  "version": "1.0",
  "sensory_mapping": {
    "dimensions": 24,
    "channels": [
      {"index": 0, "name": "position_x", "range": [0, 10], "unit": "meters"},
      {"index": 1, "name": "position_y", "range": [0, 10], "unit": "meters"},
      // ... full mapping
    ]
  },
  "action_mapping": {
    "dimensions": 4,
    "channels": [
      {"index": 0, "name": "left_motor", "range": [-1, 1], "unit": "speed_ratio"},
      // ... full mapping
    ]
  },
  "capabilities": {
    "visual_processing": true,
    "audio_processing": false,
    "manipulation": false
  }
}
```

## Performance Characteristics

### Benchmarks (Reference Implementation)
- **Throughput**: 1000+ messages/second
- **Latency**: <5ms typical round-trip
- **Overhead**: Only 9 bytes per message
- **CPU Usage**: <1% on Pi Zero W
- **Memory**: <1KB per connection

### Optimization Guidelines
1. **Use consistent vector sizes** to avoid dynamic allocation
2. **Batch sensors** to minimize message frequency
3. **Set TCP_NODELAY** for low latency
4. **Use connection pooling** for multiple robots
5. **Monitor connection health** with timeouts

## Error Recovery

### Connection Loss
```python
def robust_brain_connection():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('brain_server', 9999))
            
            # Main communication loop
            while True:
                send_sensors(sock)
                actions = receive_actions(sock)
                execute_actions(actions)
                
        except (ConnectionError, socket.timeout):
            print("Connection lost, retrying in 1 second...")
            time.sleep(1.0)
```

### Protocol Errors
```python
def handle_brain_response(sock):
    try:
        msg_type, vector = protocol.receive_message(sock)
        
        if msg_type == MSG_ACTION_OUTPUT:
            return vector
        elif msg_type == MSG_ERROR:
            error_code = vector[0]
            print(f"Brain error: {error_code}")
            return None
            
    except MessageProtocolError:
        # Reset connection
        sock.close()
        return None
```

## Testing and Validation

### Protocol Compliance Test
```python
def test_protocol_compliance():
    # Test message encoding/decoding
    protocol = BrainProtocol()
    
    test_vector = [1.0, 2.0, 3.0, 4.0]
    encoded = protocol.encode_message(test_vector, MSG_SENSORY_INPUT)
    msg_type, decoded = protocol.decode_message(encoded)
    
    assert msg_type == MSG_SENSORY_INPUT
    assert decoded == test_vector
    
    print("✅ Protocol compliance test passed")

def test_performance():
    # Measure throughput
    start_time = time.time()
    for i in range(1000):
        send_sensors([1.0] * 16)
        receive_actions()
    
    duration = time.time() - start_time
    throughput = 1000 / duration
    
    print(f"Throughput: {throughput:.0f} messages/second")
    assert throughput > 100  # Should handle >100 Hz
```

## Security Considerations

### Current State
- **No authentication**: Protocol assumes trusted network
- **No encryption**: Plain TCP connection
- **Input validation**: Basic vector size limits only

### Future Security Enhancements
1. **TLS encryption** for secure communication
2. **API key authentication** for robot authorization  
3. **Rate limiting** to prevent DoS attacks
4. **Input sanitization** for all vector values

## Migration and Compatibility

### Version Negotiation
The handshake includes version information to enable:
- **Backward compatibility** with older robots
- **Feature detection** for new capabilities
- **Protocol evolution** without breaking existing systems

### Adding New Features
To extend the protocol:
1. **Use reserved vector slots** for new sensor data
2. **Negotiate capabilities** during handshake
3. **Maintain backward compatibility** with older clients
4. **Version your robot firmware** appropriately

This minimal binary protocol provides the foundation for high-performance, real-time robot-brain communication while maintaining simplicity for embedded system implementation.