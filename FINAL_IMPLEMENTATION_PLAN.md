# Final Implementation Plan: Multi-Stream Sensors

## Current State (Working)
- **64x48 vision on TCP** - Fits in 16KB buffers, works TODAY
- All sensors packed into single 307,212 value vector
- Brain expects synchronized data

## Target State  
- **Each sensor gets its own stream** (UDP preferred, TCP fallback)
- **MJPEG video** at 640x480 @ 15fps (Pi Zero 2 W friendly)
- Brain processes whatever arrives whenever

## Implementation Steps

### Day 1: Simplest Sensor First
Add battery voltage as separate stream (parallel to existing TCP):

```python
# Robot side - new file: stream_battery.py
import socket, struct, time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Try UDP first
while True:
    voltage = read_battery()  # 7.4
    sock.sendto(struct.pack('!f', voltage), (brain_ip, 10004))
    time.sleep(10)  # Every 10 seconds is plenty

# Brain side - new listener thread
def battery_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 10004))
    while True:
        data, _ = sock.recvfrom(64)
        voltage = struct.unpack('!f', data)[0]
        brain.update_battery(voltage)  # Async update
```

**Success criteria**: Battery data flows separately while main TCP still works

### Day 2: Ultrasonic Stream
Same pattern, higher frequency:

```python
# Robot: Send distance @ 20Hz
sock.sendto(struct.pack('!f', distance_cm), (brain_ip, 10003))

# Brain: Update async
brain.update_ultrasonic(distance_cm)
```

### Day 3: Video Stream (MJPEG)

```python
# Robot side - stream_video.py
import cv2, socket, struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, jpeg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 60])
    
    # Send if fits in UDP packet
    if len(jpeg) < 65000:
        header = struct.pack('!Q', time.time_ns())
        sock.sendto(header + jpeg.tobytes(), (brain_ip, 10002))
    
    time.sleep(0.066)  # ~15 FPS

# Brain side - decode MJPEG
def video_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 10002))
    while True:
        data, _ = sock.recvfrom(65536)
        timestamp = struct.unpack('!Q', data[:8])[0]
        jpeg_data = data[8:]
        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), 
                            cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            brain.update_vision(frame)  # 640x480 grayscale
```

### Day 4: Remove Redundancy
Once streams proven working:
- Remove sensor data from TCP vector
- Keep only motor commands on TCP
- Brain fully async

### Day 5: Handle Failures
If UDP doesn't work (firewall, network issues):
```python
# Just switch to TCP!
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((brain_ip, 10004))
# Rest stays the same
```

## Brain-Side Changes Needed

```python
class Brain:
    def __init__(self):
        # Instead of expecting synchronized vector
        self.latest_vision = None
        self.latest_ultrasonic = None
        self.latest_battery = None
        
    def update_vision(self, frame):
        """Called whenever video frame arrives"""
        self.latest_vision = frame
        
    def update_ultrasonic(self, distance):
        """Called whenever ultrasonic arrives"""
        self.latest_ultrasonic = distance
        
    def process(self, motor_feedback):
        """Use whatever sensor data is available"""
        if self.latest_vision is None:
            vision = np.zeros((480, 640))  # Default
        else:
            vision = self.latest_vision
        # Process with what we have
```

## Success Metrics

1. **It works** - Data flows, brain learns
2. **Pi Zero CPU < 50%** - Sustainable  
3. **Latency < 100ms** - Responsive
4. **Handles dropouts** - Robust

## Fallback Plan

If streams fail, we always have:
- 64x48 TCP (works today!)
- Just increase to 160x120 (still fits in buffers)
- Good enough for learning

## The Key Insight

**Don't overthink it.** 

Sensors send data to ports. Brain listens on ports. Fire and forget.

If UDP works, great! If not, TCP. Either way, separate streams are better than one giant vector.

---

**Next Action**: Implement battery stream (Day 1) and see if it works!