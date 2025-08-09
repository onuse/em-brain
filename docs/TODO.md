# TODO: Comprehensive Technical Roadmap

## Executive Summary

Two critical paths: Fix brainstem architecture issues AND optimize GPU performance.
Both are necessary for production deployment of the artificial life system.

---

## PART 1: BRAINSTEM REFACTOR (4 Weeks) 🚨 CRITICAL

### Issues Discovered (2025-08-09)
The team identified significant architectural problems in the brainstem implementation:
- Protocol mismatch (HTTP client for TCP server!)
- Thread safety bombs waiting to explode
- Magic numbers throughout (if distance < 0.05)
- Backwards reward design (robot tells brain what's good)
- Arbitrary sensor padding (16→24 for no reason)
- Synchronous blocking architecture
- Monolithic untestable code

### Week 1: Foundation & Safety ⚡ IMMEDIATE
*Goal: Eliminate critical bugs and magic numbers*

#### Configuration System (Day 1-2)
- [ ] Create `brainstem_config.yaml` with all thresholds
- [ ] Replace ALL magic numbers in code
- [ ] Add environment variable overrides
- [ ] Create profile system (default, aggressive, cautious)
```yaml
safety:
  min_obstacle_distance: 0.2  # meters
  emergency_stop_distance: 0.05
  max_cpu_temperature: 70  # celsius
```

#### Fix Thread Safety (Day 3-4)
- [ ] Replace shared variables with async queues
- [ ] Implement proper locking where needed
- [ ] Convert to async/await architecture
- [ ] Add concurrent stress tests

#### Protocol Cleanup (Day 5)
- [ ] Remove sensor padding (use actual 16 sensors)
- [ ] Remove reward signal from protocol
- [ ] Remove unused capabilities_mask
- [ ] Update protocol documentation

### Week 2: Modularization 🧬
*Goal: Break monolith into biological-inspired "nuclei"*

#### Core Infrastructure (Day 1-2)
- [ ] Implement EventBus for async message passing
- [ ] Create DependencyContainer for injection
- [ ] Build base Nucleus classes
- [ ] Add telemetry hooks

#### Individual Nuclei (Day 3-4)
- [ ] **SensoryNucleus**: Sensor fusion and normalization
- [ ] **MotorNucleus**: Motor control with safety limits
- [ ] **SafetyNucleus**: Reflexes that work without brain
- [ ] **CommunicationNucleus**: Async brain connection
- [ ] **BehavioralNucleus**: Coordinate behaviors

#### Integration (Day 5)
- [ ] Wire nuclei with BrainstemConductor
- [ ] Test inter-nucleus communication
- [ ] Verify graceful degradation

### Week 3: Comprehensive Testing 🧪
*Goal: Build test pyramid (40% unit, 30% integration, 20% system, 10% behavioral)*

#### Unit Tests (Day 1-2)
```bash
tests/unit/
├── test_protocol.py          # Message encoding/decoding
├── test_safety_limits.py     # Motor clamping, emergency stops
├── test_configuration.py     # Settings validation
└── test_nuclei.py           # Individual components
```

#### Integration Tests (Day 3-4)
```bash
tests/integration/
├── test_event_bus.py        # Message passing
├── test_connection.py       # Brain connection lifecycle
├── test_error_recovery.py   # Failure handling
└── test_full_pipeline.py    # End-to-end flow
```

#### Behavioral Tests (Day 5)
```bash
tests/behavioral/
├── test_emergence.py        # Natural behaviors arise
├── test_safety_reflexes.py  # Work without brain
├── test_learning.py         # Improvement over time
└── test_stress.py          # Edge cases
```

### Week 4: Polish & Validation 🚀
*Goal: Production readiness*

- [ ] Performance optimization (hot path analysis)
- [ ] Memory profiling (no leaks in 8-hour test)
- [ ] Documentation updates
- [ ] Safety certification tests
- [ ] Deployment guide updates

### Success Metrics
- ✅ Zero magic numbers in code
- ✅ No thread safety issues (1000-iteration stress test)
- ✅ 100% protocol corruption detected
- ✅ <10ms sensor→brain→motor latency
- ✅ 8-hour stability without memory leaks
- ✅ All safety reflexes trigger 100% of time

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