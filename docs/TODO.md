# TODO: Comprehensive Technical Roadmap

## Executive Summary

Two critical paths: Fix brainstem protocol issues AND optimize GPU performance.
Both are necessary for production deployment of the artificial life system.

---

## PART 1: BRAINSTEM FIXES (1-2 Weeks) ⚡

### Current Status (2025-08-14)
After code review, the brainstem is in better shape than documented:
- ✅ Thread safety: Single-threaded design, no race conditions
- ✅ Safety reflexes: Properly implemented with emergency stops
- ✅ Clean architecture: Well-structured Brainstem class
- ✅ No HTTP confusion: Using proper TCP sockets

### Actual Issues to Fix
1. **Protocol Mismatch** (CRITICAL - blocks all communication):
   - Client uses old protocol: 'H', 'S', 'M' character headers
   - Server expects MessageProtocol with proper binary headers
   - Must update brain_client.py to match server protocol

2. **Magic Numbers** (Code quality issue):
   - Hard-coded thresholds: 10.0cm collision, 3500 ADC cliff, 6.0V battery
   - Hard-coded conversions: distance/58.0, battery*10.0
   - Fixed resolution: 307200 pixels (640x480)

3. **No Configuration System**:
   - All values hard-coded in SafetyConfig dataclass
   - No YAML config file
   - No environment variable overrides

4. **Synchronous Architecture** (Performance limitation):
   - Blocking operations limit responsiveness
   - No async/await for concurrent operations

### Week 1: Critical Fixes 🚨
*Goal: Fix protocol mismatch and eliminate magic numbers*

#### Day 1-2: Fix Protocol Mismatch (BLOCKER)
- [ ] Update brain_client.py to use MessageProtocol
- [ ] Test client-server communication
- [ ] Verify handshake and data exchange
- [ ] Update any other clients using old protocol

#### Day 3-4: Configuration System
- [ ] Create `brainstem_config.yaml` with all thresholds
- [ ] Implement ConfigurationManager to load YAML
- [ ] Replace ALL magic numbers in code
- [ ] Add environment variable overrides
```yaml
safety:
  collision_distance_cm: 10.0
  cliff_threshold_adc: 3500
  battery_critical_v: 6.0
  max_temp_c: 80.0
conversions:
  ultrasonic_us_to_cm: 58.0
  battery_adc_to_volts: 10.0
  vision_resolution: [640, 480]
```

#### Day 5: Testing
- [ ] Unit tests for protocol compatibility
- [ ] Integration test for brain-robot communication
- [ ] Safety reflex tests
- [ ] Configuration loading tests

### Week 2: Optional Improvements 🔧
*Only if time permits - not critical for deployment*

#### Performance (Optional)
- [ ] Convert to async/await architecture
- [ ] Add connection pooling
- [ ] Implement message batching
- [ ] Profile and optimize hot paths

#### Code Quality (Optional)
- [ ] Add comprehensive logging/telemetry
- [ ] Create mock HAL for testing
- [ ] Add type hints throughout
- [ ] Document all public APIs

### Removed from Scope
The following were in the original plan but are not needed:
- ❌ Thread safety fixes (no threading issues exist)
- ❌ Biological nuclei architecture (over-engineering)
- ❌ EventBus pattern (unnecessary complexity)
- ❌ 4-week timeline (1-2 weeks sufficient)

### Success Metrics
- ✅ Protocol compatibility restored (client-server communication works)
- ✅ Zero magic numbers in code (all in config)
- ✅ Safety reflexes trigger 100% of time
- ✅ <50ms sensor→brain→motor latency (current synchronous design)
- ✅ 8-hour stability without crashes
- ✅ Configuration system with YAML + env vars

---

## PART 2: GPU OPTIMIZATION (5 Weeks) 🎮

### Current Performance
- CPU: 100ms/cycle (too slow for real-time)
- Target: <1ms/cycle on GPU (100x+ speedup)

### Phase 0: Foundation Cleanup (1 week)
*Prepare codebase for GPU migration*

#### Memory Architecture Refactor
- [ ] **Unify tensor allocation** (Priority: CRITICAL)
  - Remove all `torch.zeros()` from hot paths
  - Pre-allocate all tensors at startup
  - Use views instead of new allocations
  
#### Remove CPU-bound Operations
- [ ] **Eliminate NumPy conversions** (Priority: CRITICAL)
  - Audit all `.cpu().numpy()` calls
  - Replace with pure PyTorch operations
  - Keep everything on GPU until final output

#### Measurement Infrastructure
- [ ] **Add GPU profiling hooks**
  - Profile every major operation
  - Identify kernel bottlenecks
  - Track memory bandwidth usage

### Phase 1: Core GPU Migration (2 weeks)
*Move critical path entirely to GPU*

#### Unified Field Dynamics Kernel
- [ ] **Single fused kernel for field evolution**
  ```python
  @torch.jit.script
  def evolve_field_fused(field, sensory, prediction_error):
      # All operations in single kernel
      # Target: <0.1ms for full evolution
  ```

#### Gradient Extraction Optimization
- [ ] **Vectorized gradient computation**
  - Use 3D Sobel filters
  - Replace nested loops with convolutions
  - Target: <0.05ms

#### Pattern Operations on GPU
- [ ] **Pattern library as texture memory**
  - Parallel similarity computation
  - GPU-based pattern blending

### Phase 2: Advanced GPU Optimizations (1 week)
*Squeeze maximum performance*

#### Memory Bandwidth Optimization
- [ ] **Optimize tensor layout**
  - Change to channels-first: [Features, X, Y, Z]
  - Ensure coalesced memory access
  - Target: 80% bandwidth efficiency

#### Custom CUDA Kernels (if needed)
- [ ] **Write custom kernels for bottlenecks**
  - Only if PyTorch insufficient
  - Focus on innermost loops
  - Target: Additional 2-5x speedup

#### Multi-Stream Execution
- [ ] **Overlap compute and memory transfers**
  - Concurrent sensory input transfer
  - Parallel field evolution
  - Async motor output

### Phase 3: Real Robot Integration (1 week)
*Ensure GPU brain works with hardware*

#### Latency-Optimized Pipeline
- [ ] **Zero-copy sensor integration**
  - Pre-allocate pinned memory
  - Direct GPU transfer
  - Non-blocking operations

#### Adaptive Compute Budget
- [ ] **Dynamic quality scaling**
  - Reduce quality if over budget
  - Increase quality if under budget
  - Maintain real-time guarantee

### Performance Targets
- **Cycle time**: <1ms on RTX 4090, <5ms on Jetson Orin
- **Memory usage**: <4GB GPU RAM
- **Zero allocations** during steady state
- **Response time**: <10ms sensor to motor

---

## PART 3: DOCUMENTATION CLEANUP 📚

### Move to docs/ folder
- [ ] BRAIN_ASSESSMENT_2025.md → docs/assessments/
- [ ] REFACTOR_AND_TEST_PLAN.md → docs/planning/
- [ ] ROBOT_DEPLOYMENT_GUIDE.md → docs/deployment/
- [ ] GPU optimization docs → docs/optimization/
- [ ] Architecture docs → docs/architecture/

### Delete redundant files
- [ ] Remove duplicate analysis files
- [ ] Clean up old summaries
- [ ] Archive obsolete plans

### Update critical docs
- [ ] README.md - Current project status
- [ ] CLAUDE.md - Latest development guidance
- [ ] COMM_PROTOCOL.md - Fixed protocol spec

---

## PART 4: TESTING STRATEGY 🧪

### Testing Pyramid Implementation
```
         Behavioral Tests (10%)
              /       \
       System Tests (20%)     
           /       \
     Integration Tests (30%)
         /       \
    Unit Tests (40%)
```

### Critical Test Suites

#### Safety-Critical Tests (MANDATORY before deployment)
- [ ] Collision prevention (<100ms response)
- [ ] Cliff avoidance (<50ms response)
- [ ] Emergency stop (<20ms response)
- [ ] Battery protection
- [ ] Temperature protection

#### Behavioral Tests
- [ ] Emergence verification (behaviors arise naturally)
- [ ] Learning progression (improvement over time)
- [ ] Field coherence (0.1-2.0 variance)
- [ ] No pathological behaviors

#### Performance Tests
- [ ] Latency benchmarks
- [ ] Throughput tests
- [ ] Memory leak detection
- [ ] 8-hour stability test

---

## Priority Order 📊

### 🔴 CRITICAL (This Week)
1. Fix thread safety issues
2. Create configuration system
3. Remove magic numbers
4. Fix protocol issues

### 🟡 HIGH (Next 2 Weeks)
5. Modularize into nuclei
6. Build test suite
7. Start GPU foundation cleanup

### 🟢 MEDIUM (Weeks 3-4)
8. Complete GPU migration
9. Documentation cleanup
10. Performance optimization

### 🔵 LOW (Future)
11. Custom CUDA kernels
12. Advanced telemetry
13. Visual monitoring dashboard

---

## Team Assignments 👥

### Pragmatic Engineer
- Configuration system
- Performance optimization
- Unit test framework

### Cognitive Analyst
- Modular architecture design
- Event bus implementation
- Integration tests

### Behavioral Scientist
- Behavioral test suite
- Emergence metrics
- Safety validation

### GPU Architect
- GPU migration strategy
- Kernel optimization
- Performance benchmarks

---

## Success Criteria ✅

### Brainstem Refactor
- All critical bugs fixed
- Modular, testable architecture
- Comprehensive test coverage
- No magic numbers
- Thread-safe async design

### GPU Optimization
- 100x speedup achieved
- <1ms cycle time
- Real-time robot control
- No memory leaks

### Overall System
- Safe for physical deployment
- Emergent behaviors preserved
- 8-hour stability demonstrated
- Production-ready code

---

## Next Actions (TODAY) 🎯

1. **Start configuration system** - Eliminate magic numbers
2. **Fix thread safety** - Prevent race conditions
3. **Create unit tests** - Protocol and safety
4. **Document current issues** - Update this TODO as we progress

---

*"Test the infrastructure thoroughly, let the intelligence emerge freely."*

Last Updated: 2025-08-09
Status: ACTIVE DEVELOPMENT
Confidence: HIGH (with proper testing)