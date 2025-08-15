# Video Streaming Strategy for Pi Zero 2 W

## The Reality
Pi Zero 2 W has **no hardware video encoder**. H.264 encoding would kill the CPU.

## The Solution: MJPEG

### Why MJPEG?
1. **Each frame independent** - Lose one? No problem! No keyframe dependencies
2. **Camera does JPEG compression** - Pi Camera has hardware JPEG
3. **Simple to implement** - It's just sending JPEG images
4. **Perfect for UDP** - No codec state to maintain

### Implementation

```python
import cv2
import socket
import struct
import time

class MJPEGStreamer:
    def __init__(self, brain_ip, port=10002):
        self.brain_ip = brain_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Configure camera for MJPEG
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # 15fps is plenty
        
    def stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Encode as JPEG (hardware accelerated on Pi Camera)
            _, jpeg = cv2.imencode('.jpg', frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, 60])
            
            # Simple header: timestamp + frame size
            timestamp = int(time.time() * 1e6)
            header = struct.pack('!QI', timestamp, len(jpeg))
            
            # Send as single UDP packet if small enough
            if len(jpeg) < 60000:  # Safe UDP size
                packet = header + jpeg.tobytes()
                self.sock.sendto(packet, (self.brain_ip, self.port))
            else:
                # Fragment if needed (rare at 640x480)
                # Or just skip this frame
                pass
```

### Brain Side

```python
class MJPEGReceiver:
    def __init__(self, port=10002):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        self.sock.settimeout(0.001)
        
    def receive_frames(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
                
                # Parse header
                timestamp, size = struct.unpack('!QI', data[:12])
                jpeg_data = data[12:12+size]
                
                # Decode JPEG
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8), 
                    cv2.IMREAD_GRAYSCALE
                )
                
                if frame is not None:
                    yield frame
                # Missing frame? Brain predicts!
                
            except socket.timeout:
                yield None  # No frame available
```

## Bandwidth Calculation

### 640x480 @ 15fps with JPEG quality 60
- ~30KB per frame
- 15 fps × 30KB = 450KB/s = 3.6 Mbps
- **Easily handles by Pi Zero WiFi!**

### Comparison
- H.264: Would need CPU encoding (impossible on Zero)
- Raw 640x480: 9.2 Mbps (higher than MJPEG!)
- MJPEG: 3.6 Mbps with hardware JPEG

## Progressive Enhancement Path

### Start: 320x240 @ 15fps
- ~10KB per frame = 1.2 Mbps
- Definitely works

### Test: 640x480 @ 15fps  
- ~30KB per frame = 3.6 Mbps
- Should work well

### Dream: 1280x720 @ 10fps
- ~80KB per frame = 6.4 Mbps
- Might work on good WiFi

## Key Advantages for Brain Learning

1. **Frame independence** - Every frame standalone
2. **Temporal gaps OK** - Missing frames create prediction pressure
3. **Variable framerate** - Brain learns real timing, not fixed rate
4. **Graceful degradation** - Lower quality/resolution when needed

## The Verdict

**MJPEG is PERFECT for Pi Zero 2 W:**
- Hardware JPEG compression (in camera)
- No CPU encoding needed
- Frame independence matches UDP philosophy
- Good enough quality for learning
- Simple to implement

**Forget H.264 on Pi Zero 2 W. Use MJPEG!**