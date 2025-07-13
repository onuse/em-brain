# Protocol Implementation Details Addendum

**For Third-Party Brainstem Implementers**

This document covers implementation details not fully specified in the main protocol documentation.

## **Critical Constants and Limits**

### **Vector Size Limits**
```python
MAX_VECTOR_SIZE = 1024  # Safety limit enforced by server
```
- **Recommended sensory vector size**: 16 floats
- **Recommended action vector size**: 4 floats  
- **Maximum allowed**: 1024 floats per message
- **Behavior if exceeded**: Server returns ERROR message with code 4.0

### **Timeout Values (Implementation Details)**
```python
# Client-side timeouts
CONNECTION_TIMEOUT = 10.0      # Socket connection to server
HANDSHAKE_TIMEOUT = 5.0        # Handshake response wait
ACTION_REQUEST_TIMEOUT = 1.0   # Default action request timeout

# Server-side timeouts  
CLIENT_SESSION_TIMEOUT = 30.0  # Client session keep-alive
MESSAGE_RECEIVE_TIMEOUT = 10.0 # Individual message receive
```

## **Handshake Protocol Details**

### **Default Robot Capabilities (Client)**
When no capabilities specified, client sends:
```python
robot_capabilities = [
    1.0,   # Robot software version
    16.0,  # Sensory vector size 
    4.0,   # Action vector size
    1.0    # Hardware type (1.0 = PiCar-X)
]
```

### **Server Capabilities Response**
Server always responds with:
```python
server_capabilities = [
    4.0,   # Brain software version  
    16.0,  # Expected sensory vector size
    4.0,   # Action vector size
    1.0    # GPU acceleration available (1.0=yes, 0.0=no)
]
```

## **Connection Behavior Details**

### **Server Connection Handling**
- **Bind address**: `0.0.0.0:9999` (all interfaces)
- **Client limit**: No hardcoded limit (system dependent)
- **Thread model**: One thread per client connection
- **Keep-alive**: 30-second client timeout, automatic disconnect

### **Client Connection Recovery**
```python
# Robust connection pattern (recommended)
def connect_with_retry(max_retries=5, retry_delay=1.0):
    for attempt in range(max_retries):
        try:
            client = MinimalBrainClient()
            if client.connect():
                return client
        except ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
```

## **Error Code Details**

### **Extended Error Codes (Implementation)**
```python
ERROR_CODES = {
    1.0: "Unknown message type",
    2.0: "Protocol error",  
    3.0: "Empty sensory input",
    4.0: "Sensory input too large (>1024 elements)",
    5.0: "Brain processing error",
    6.0: "Client timeout",           # Server-side timeout
    7.0: "Handshake failed",         # Incompatible versions
    8.0: "Server overloaded"         # Too many clients
}
```

## **Performance Tuning**

### **TCP Socket Options (Recommended)**
```python
# For low-latency real-time communication
socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

# For embedded systems with limited memory
socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
```

### **Message Frequency Guidelines**
- **Typical robot control**: 10-50 Hz (20-100ms between messages)
- **High-performance**: Up to 100 Hz (10ms between messages)  
- **Embedded systems**: 5-20 Hz (50-200ms between messages)
- **Network consideration**: Batch multiple sensor readings if needed

## **Development and Testing**

### **Protocol Validation**
```python
def validate_protocol_compliance():
    """Test basic protocol compliance"""
    # Test 1: Message encoding/decoding
    protocol = MessageProtocol()
    test_vector = [1.0, 2.0, 3.0, 4.0]
    
    encoded = protocol.encode_sensory_input(test_vector)
    msg_type, decoded = protocol.decode_message(encoded)
    
    assert msg_type == MessageProtocol.MSG_SENSORY_INPUT
    assert decoded == test_vector
    
    # Test 2: Size limits
    try:
        large_vector = [1.0] * 2000  # Too large
        protocol.encode_sensory_input(large_vector)
        assert False, "Should have failed"
    except ValueError:
        pass  # Expected
    
    print("✅ Protocol compliance validated")
```

### **Connection Testing**
```python
def test_brain_connectivity(host='localhost', port=9999):
    """Test basic brain server connectivity"""
    try:
        client = MinimalBrainClient(host, port)
        if client.connect():
            # Send test vector
            test_sensors = [0.0] * 16
            actions = client.get_action(test_sensors, timeout=2.0)
            if actions and len(actions) == 4:
                print(f"✅ Brain connectivity test passed: {actions}")
                return True
    except Exception as e:
        print(f"❌ Brain connectivity test failed: {e}")
    return False
```

## **Platform-Specific Notes**

### **Raspberry Pi / Embedded Systems**
- **Memory usage**: ~1KB per active connection
- **CPU usage**: <1% on Pi Zero W at 10Hz
- **Network buffer**: Use smaller socket buffers (4-8KB) to reduce memory
- **Threading**: Single-threaded client recommended for embedded systems

### **Arduino / Microcontrollers**
- **WiFi modules**: ESP32, ESP8266 can handle the protocol
- **Memory requirements**: ~2KB RAM for protocol implementation  
- **Float precision**: Ensure IEEE 754 float32 compatibility
- **Endianness**: Use network byte order (big-endian)

### **Real-Time Systems**
- **Deterministic timing**: Account for network latency variance
- **Timeout handling**: Use shorter timeouts for real-time requirements
- **Priority**: Run protocol communication in high-priority thread
- **Jitter management**: Consider message queuing for smooth control

## **Integration Examples**

### **ROS Integration**
```python
# Example ROS node integration
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class BrainROSBridge:
    def __init__(self):
        self.client = MinimalBrainClient()
        self.client.connect()
        
        # ROS publishers/subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.sensor_callback)
    
    def sensor_callback(self, joint_msg):
        # Convert ROS sensor data to brain vector
        sensors = self.ros_to_brain_vector(joint_msg)
        actions = self.client.get_action(sensors)
        
        # Convert brain actions to ROS commands
        cmd_vel = self.brain_to_ros_twist(actions)
        self.cmd_pub.publish(cmd_vel)
```

### **Custom Hardware Integration**
```python
# Example for custom robot hardware
class CustomBrainstem:
    def __init__(self):
        self.brain_client = MinimalBrainClient()
        self.hardware = CustomHardwareInterface()
        
    def control_loop(self):
        while True:
            # Read all sensors
            sensors = [
                self.hardware.get_position_x(),
                self.hardware.get_position_y(), 
                self.hardware.get_heading(),
                self.hardware.get_speed(),
                # ... 12 more sensor values
            ]
            
            # Get brain decision
            actions = self.brain_client.get_action(sensors)
            
            # Execute actions
            self.hardware.set_left_motor(actions[0])
            self.hardware.set_right_motor(actions[1]) 
            self.hardware.set_steering(actions[2])
            self.hardware.set_camera_pan(actions[3])
            
            time.sleep(0.05)  # 20Hz control loop
```

## **Troubleshooting Guide**

### **Common Issues**

1. **"Connection refused"**
   - Check brain server is running on correct port
   - Verify firewall settings
   - Confirm network connectivity

2. **"Message too large" errors**
   - Ensure sensor vector ≤ 1024 elements
   - Check for data corruption in vector creation
   - Validate float values aren't NaN/infinite

3. **Timeout errors**
   - Increase timeout values for slower networks
   - Check for blocking operations in sensor reading
   - Monitor server load and performance

4. **Inconsistent actions**
   - Verify sensor vector consistency (same size, order)
   - Check for sensor noise/outliers
   - Monitor brain learning progress

### **Debug Protocol Messages**
```python
def debug_protocol_messages(enable_logging=True):
    """Enable detailed protocol message logging"""
    if enable_logging:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Log all messages
        original_send = MessageProtocol.send_message
        def debug_send(self, sock, data):
            info = self.get_message_info(data)
            logging.debug(f"SEND: {info}")
            return original_send(self, sock, data)
        MessageProtocol.send_message = debug_send
```

---

This addendum provides the implementation details needed for robust third-party brainstem development while maintaining compatibility with the core brain architecture.