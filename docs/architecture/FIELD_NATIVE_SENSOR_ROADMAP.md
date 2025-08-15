# Field-Native Parallel Sensor Architecture Roadmap

## Executive Summary

Move from synchronized sensor processing to parallel field injection, where each sensor stream gets its own processing thread that writes directly to designated field regions. This eliminates the visual processing bottleneck and mirrors biological neural architecture.

## The Problem

Current architecture bottleneck:
```
All sensors → Buffer → Main thread processes → Field → Motor
                        ↑
                  [BLOCKING on vision!]
```

Vision processing (307,200 pixels → features) can consume 98% of cycle time, blocking all other sensors and motor output. This is fundamentally wrong.

## The Solution

Parallel field injection:
```
Vision Thread    → Visual cortex region     ┐
Ultrasonic Thread → Distance region          ├→ Field → Motor
IMU Thread       → Proprioceptive region    ┘
Battery Thread   → Homeostatic region

Main thread just evolves field and extracts motor!
```

## Why This Is Right

### Biological Correctness
- Real brains have parallel sensory processing
- Visual cortex processes continuously, doesn't block hearing
- No global synchronization in biology
- Conflicts resolved through field dynamics, not locks

### Engineering Benefits
- Vision processing doesn't block motor commands
- Each sensor at optimal rate (IMU 100Hz, Vision 30Hz, etc)
- Natural load balancing across CPU cores
- Graceful degradation (one sensor fails, others continue)

### Already Compatible
- Brain only reads field state, not buffers
- Field evolution naturally integrates parallel inputs
- No changes needed to core brain logic
- Just changes WHERE data enters field

## Implementation Phases

### Phase 1: Regional Injection (Week 1)
Keep existing buffer system but inject into specific regions:

```python
# Before: All sensors → combined vector → whole field
# After: Each sensor → specific region

sensor_regions = {
    'vision': field[0:8, 0:8, 0:8, 0:32],      # Visual cortex
    'ultrasonic': field[8:10, 0:2, 0:2, 32:36], # Distance
    'imu': field[4:8, 4:8, 4:8, 36:42],         # Proprioception
    'battery': field[0:1, 0:1, 0:1, 60:61]      # Homeostasis
}
```

**Goal**: Prove regional injection works without breaking learning

### Phase 2: Direct Injection Threads (Week 2)
Skip buffers for simple sensors:

```python
def ultrasonic_thread():
    """Direct field injection, no buffer"""
    sock = bind(10003)
    while True:
        distance = sock.recv()
        field[distance_region] = process_distance(distance)
        # No waiting for main loop!
```

**Goal**: Prove parallel injection doesn't break field coherence

### Phase 3: Vision Processing Thread (Week 3)
Move heavy vision processing off main thread:

```python
def vision_thread():
    """Heavy processing in dedicated thread"""
    sock = bind(10002)
    while True:
        jpeg = sock.recv()
        frame = decode_jpeg(jpeg)           # CPU intensive
        edges = extract_edges(frame)        # CPU intensive  
        features = extract_features(edges)  # CPU intensive
        field[visual_cortex] = features     # Quick injection
```

**Goal**: Main thread stays at 50Hz even with vision processing

### Phase 4: Dynamic Thread Spawning (Week 4)
During handshake, spawn threads based on declared sensors:

```python
def handshake(robot_capabilities):
    """Spawn thread for each declared sensor stream"""
    for sensor in robot_capabilities['sensors']:
        if sensor['type'] == 'vision':
            spawn_vision_thread(sensor['port'])
        elif sensor['type'] == 'ultrasonic':
            spawn_ultrasonic_thread(sensor['port'])
        # ... etc
    
    # Main thread just does field evolution
    return assigned_ports
```

**Goal**: Automatic configuration based on robot capabilities

## Technical Considerations

### Thread Safety
```python
# Option 1: Let conflicts happen (biological)
field[region] += data  # Occasional races = neural noise

# Option 2: Regional locks (if needed)
with region_locks[sensor]:
    field[region] = data

# Option 3: Atomic operations
field.index_add_(region, data)  # PyTorch atomic
```

Recommendation: Start with Option 1 (biological noise)

### Field Coherence
- Field diffusion naturally blends adjacent regions
- Decay prevents runaway activation
- Evolution continues regardless of injection timing

### Performance Monitoring
Track per-thread metrics:
- Processing time per sensor
- Injection frequency
- Field region activation levels
- Cross-region information flow

## Migration Strategy

### Week 1: Preparation
- [x] Implement AsyncBrainAdapter (done!)
- [ ] Define sensor regions in field
- [ ] Test regional injection with existing system

### Week 2: Simple Sensors
- [ ] Ultrasonic direct injection thread
- [ ] Battery direct injection thread
- [ ] Verify field coherence maintained

### Week 3: Vision Thread
- [ ] Move vision processing to thread
- [ ] Benchmark main loop performance
- [ ] Tune injection rates

### Week 4: Full System
- [ ] Dynamic thread spawning
- [ ] All sensors parallel
- [ ] Performance validation

## Success Metrics

### Performance
- Main loop maintains 50Hz with vision processing
- Vision processing doesn't block motor commands
- CPU utilization across multiple cores
- Latency from sensor to motor < 50ms

### Behavioral
- Learning continues normally
- No degradation in navigation
- Emergence still occurs
- Robustness to sensor dropouts

## Risk Mitigation

### Risk: Field incoherence from parallel writes
**Mitigation**: Start with non-overlapping regions, gradual overlap

### Risk: Non-deterministic behavior
**Mitigation**: This is actually biological! But can add optional seeds for debugging

### Risk: Hard to debug
**Mitigation**: Thread activity visualization, field heatmaps, extensive logging

## Code Example: Full Vision Thread

```python
class VisionProcessingThread:
    def __init__(self, brain_field, port=10002):
        self.field = brain_field
        self.visual_cortex = (slice(0,8), slice(0,8), slice(0,8), slice(0,32))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        
    def run(self):
        while True:
            # Receive MJPEG frame
            data, _ = self.sock.recvfrom(65536)
            
            # Heavy processing (doesn't block main thread!)
            frame = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            
            # Extract features (expensive!)
            edges = cv2.Canny(frame, 100, 200)
            corners = cv2.goodFeaturesToTrack(frame, 25, 0.01, 10)
            optical_flow = self.compute_flow(frame)
            
            # Combine into field tensor
            features = self.features_to_tensor(edges, corners, optical_flow)
            
            # Direct injection (no buffer, no main thread!)
            with torch.no_grad():
                self.field[self.visual_cortex] *= 0.95  # Decay
                self.field[self.visual_cortex] += features * 0.1  # Inject
            
            # Vision processing complete, main thread never blocked!
```

## Philosophical Alignment

This architecture embraces:
- **Biological realism**: Parallel sensory processing like real brains
- **Emergent complexity**: Simple parallel inputs → complex behaviors
- **Robustness**: No single point of failure
- **Efficiency**: Use all CPU cores, not just one

## Next Steps

1. Review and approve this roadmap
2. Start Phase 1: Define field regions
3. Test regional injection with battery sensor
4. Gradually migrate sensors to parallel processing

---

**The key insight: The brain already only cares about field state. We're just changing HOW data gets into the field, not how the brain processes it.**

*"Real brains don't wait for vision to finish before processing sound. Neither should ours."*