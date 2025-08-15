# Implementation Roadmap: TCP → UDP Streams

## Current State (Working!)
- **64x48 vision on TCP** - Works today, no fragmentation
- All sensors packed into single control message
- Brain receives everything synchronously

## Target State
- **MJPEG video stream over UDP** - 640x480 @ 15fps (Pi Zero 2 W friendly!)
- Each sensor on its own UDP port
- Brain processes asynchronously, embraces messiness

## Implementation Steps

### Week 1: Proof of Concept
**Monday**: Test current 64x48 system thoroughly
- Verify robot-brain communication works
- Document baseline behavior
- Ensure brain learns from low-res vision

**Tuesday**: Add first UDP sensor (ultrasonic)
```python
# Robot side - parallel to existing TCP
sock.sendto(struct.pack('!f', distance), (brain_ip, 10003))

# Brain side - new UDP listener thread
udp_sock.bind(('', 10003))
distance = struct.unpack('!f', udp_sock.recv(4))[0]
```

**Wednesday**: Test with packet loss
- Introduce artificial drops (10%, 20%, 30%)
- Verify brain still functions
- Document adaptation behaviors

### Week 2: Video Streaming
**Thursday**: Implement MJPEG streaming
```python
# Robot: Stream MJPEG (Pi Zero 2 W has no H.264 encoder!)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
_, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
sock.sendto(jpeg.tobytes(), (brain_ip, 10002))

# Brain: Decode JPEG frames
frame = cv2.imdecode(jpeg_data, cv2.IMREAD_GRAYSCALE)
```

**Friday**: Integration testing
- Run both TCP control + UDP streams
- Verify no interference
- Test with HD resolution

### Week 3: Full Migration
**Monday**: Add remaining sensors to UDP
- IMU → UDP:10001 (100Hz)
- Battery → UDP:10004 (0.1Hz)

**Tuesday**: Remove sensor data from TCP
- Keep only motor commands on TCP
- Clean separation of concerns

**Wednesday-Friday**: Validation
- Extended testing with full system
- Document emergent behaviors
- Performance measurements

## Success Criteria

### Technical
- [ ] MJPEG video streaming works (640x480 @ 15fps)
- [ ] <50ms latency for motor commands
- [ ] Brain handles 30% packet loss gracefully
- [ ] No memory leaks over 8 hours
- [ ] Pi Zero 2 W CPU stays under 50%

### Behavioral  
- [ ] Navigation behaviors emerge
- [ ] Learning improves over time
- [ ] Robust to sensor dropouts
- [ ] Temporal patterns learned

## Key Decisions Made

1. **All sensors use UDP** (except motors)
   - Embraces biological messiness
   - No synchronization needed
   - Fire-and-forget simplicity

2. **Video as MJPEG stream** (not H.264!)
   - Pi Zero 2 W has NO H.264 encoder
   - Hardware JPEG from camera module
   - Frame independence perfect for UDP

3. **Single robot focus**
   - No multi-robot complexity
   - Hardcoded ports (10001-10099)
   - Local network only

4. **64x48 is temporary**
   - Just a TCP buffer workaround
   - Will be replaced by HD streaming
   - Good enough to start learning NOW

## Philosophy

**"In biology, messy solutions that work beat perfect solutions that don't exist."**

- Packet loss is good (builds robustness)
- Jitter is natural (like neural timing)
- Async is biological (no central clock)
- Simple beats complex (always)

## Next Action

```bash
# Test current 64x48 system
cd client_picarx
python3 picarx_robot.py --brain-host <SERVER-IP>

# Brain learns from whatever it gets
# That's the beauty of biological messiness
```

---

*Remember: The fruit fly navigates with 800 pixels. Your robot has 3,072 now, and will have 2,073,600 soon. Both are enough to learn.*