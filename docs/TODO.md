# TODO: Technical Roadmap - GPU Optimization Critical Path

## Executive Summary

Current system: 100ms/cycle on CPU → Target: <1ms/cycle on GPU (100x+ speedup)
While preserving: Field-native intelligence, emergent behaviors, real robot compatibility

## Phase 0: Foundation Cleanup (1 week)
*Prepare codebase for GPU migration*

### Memory Architecture Refactor
- [ ] **Unify tensor allocation** (Priority: CRITICAL)
  ```python
  # Current: scattered allocations across 15+ locations
  # Target: single managed pool with views
  class FieldMemoryPool:
      def __init__(self, device='cuda'):
          # Pre-allocate all tensors at startup
          self.field = torch.zeros(32, 32, 32, 64, device=device)
          self.gradients = torch.zeros_like(self.field)
          self.patterns = torch.zeros(100, 32, 32, 32, 16, device=device)
          # Return views, never allocate in hot path
  ```
  - Remove all `torch.zeros()` calls from hot paths
  - Pre-allocate work buffers for all operations
  - **Success metric**: Zero allocations during steady-state operation

### Remove CPU-bound Operations
- [ ] **Eliminate NumPy conversions** (Priority: CRITICAL)
  - Audit all `.cpu().numpy()` calls
  - Replace with pure PyTorch operations
  - Keep everything on GPU until final motor output
  - **Files to fix**: 
    - `unified_field_brain.py`: Lines with numpy conversions
    - `evolved_field_dynamics.py`: Statistical calculations
    - `motor_cortex.py`: Final output generation

- [ ] **Replace Python loops with tensor operations** (Priority: HIGH)
  ```python
  # Before: 
  for i in range(32):
      for j in range(32):
          field[i,j] = compute_value(i, j)
  
  # After:
  indices = torch.meshgrid(torch.arange(32), torch.arange(32))
  field = compute_value_vectorized(indices)
  ```

### Measurement Infrastructure
- [ ] **Add GPU profiling hooks** (Priority: HIGH)
  ```python
  class GPUProfiler:
      def profile_kernel(self, name, func, *args):
          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)
          start.record()
          result = func(*args)
          end.record()
          torch.cuda.synchronize()
          self.timings[name] = start.elapsed_time(end)
          return result
  ```
  - Profile every major operation
  - Identify GPU kernel bottlenecks
  - Track memory bandwidth usage

## Phase 1: Core GPU Migration (2 weeks)
*Move critical path entirely to GPU*

### Unified Field Dynamics Kernel
- [ ] **Single fused kernel for field evolution** (Priority: CRITICAL)
  ```python
  @torch.jit.script
  def evolve_field_fused(field, sensory, prediction_error, patterns):
      # All operations in single kernel - no intermediate tensors
      # 1. Diffusion (3D convolution)
      diffused = F.conv3d(field.unsqueeze(0), diffusion_kernel)
      
      # 2. Activation dynamics (tanh, keeping values bounded)
      activated = torch.tanh(diffused * 0.9)
      
      # 3. Sensory integration (additive with decay)
      integrated = activated * 0.95 + sensory * 0.05
      
      # 4. Prediction error learning (in-place update)
      learned = integrated + prediction_error * learning_rate
      
      # 5. Pattern influence (strategic channels)
      field[:,:,:,32:48] = learned[:,:,:,32:48] * 0.9 + patterns * 0.1
      
      return learned
  ```
  - **Target**: <0.1ms for full field evolution
  - No intermediate allocations
  - All operations fused into single kernel

### Gradient Extraction Optimization
- [ ] **Vectorized gradient computation** (Priority: CRITICAL)
  ```python
  def extract_gradients_gpu(field):
      # Use 3D Sobel filters for all gradients at once
      sobel_x = torch.tensor([[[[-1,0,1]...]]], device='cuda')
      grad_x = F.conv3d(field.permute(3,0,1,2).unsqueeze(1), sobel_x)
      grad_y = F.conv3d(field.permute(3,0,1,2).unsqueeze(1), sobel_y)
      grad_z = F.conv3d(field.permute(3,0,1,2).unsqueeze(1), sobel_z)
      
      # Combine into final gradient tensor - no loops!
      gradients = torch.cat([grad_x, grad_y, grad_z], dim=1)
      return gradients.permute(2,3,4,0,1)
  ```
  - Replace nested loops with convolutions
  - **Target**: <0.05ms for gradient extraction

### Pattern Operations on GPU
- [ ] **Pattern library as texture memory** (Priority: HIGH)
  ```python
  class GPUPatternLibrary:
      def __init__(self, max_patterns=100):
          # Store patterns in optimal layout for GPU access
          self.patterns = torch.zeros(max_patterns, 32*32*32*16, device='cuda')
          self.contexts = torch.zeros(max_patterns, 64, device='cuda')
          
      def retrieve_pattern_batch(self, query_contexts):
          # Compute all similarities in parallel
          similarities = torch.matmul(query_contexts, self.contexts.T)
          top_k = torch.topk(similarities, k=3, dim=1)
          # Blend patterns using GPU operations
          return self.blend_patterns_gpu(top_k.indices, top_k.values)
  ```

## Phase 2: Advanced GPU Optimizations (1 week)
*Squeeze maximum performance from GPU*

### Memory Bandwidth Optimization
- [ ] **Optimize tensor layout** (Priority: HIGH)
  ```python
  # Current layout: [X, Y, Z, Features]
  # Optimal for GPU: [Features, X, Y, Z] (channels-first)
  # This matches CUDA's memory coalescing patterns
  
  field = field.permute(3, 0, 1, 2).contiguous()
  ```
  - Ensure coalesced memory access
  - Use texture memory for read-only patterns
  - **Target**: 80% memory bandwidth efficiency

### Custom CUDA Kernels (if needed)
- [ ] **Write custom kernels for bottlenecks** (Priority: MEDIUM)
  ```cuda
  __global__ void evolve_field_kernel(
      float* field, float* sensory, float* output,
      int x_dim, int y_dim, int z_dim, int f_dim
  ) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < x_dim * y_dim * z_dim * f_dim) {
          // Custom optimized evolution logic
          // Use shared memory for neighborhood operations
          // Minimize divergent branches
      }
  }
  ```
  - Only if PyTorch operations prove insufficient
  - Focus on innermost loops
  - **Target**: Additional 2-5x speedup for critical operations

### Multi-Stream Execution
- [ ] **Overlap compute and memory transfers** (Priority: MEDIUM)
  ```python
  class StreamedBrain:
      def __init__(self):
          self.stream_compute = torch.cuda.Stream()
          self.stream_memory = torch.cuda.Stream()
          
      def process(self, sensory_input):
          with torch.cuda.stream(self.stream_memory):
              # Transfer next sensory input while computing
              next_sensory = sensory_input.to('cuda', non_blocking=True)
          
          with torch.cuda.stream(self.stream_compute):
              # Process current field state
              self.field = evolve_field_fused(self.field, self.sensory)
          
          torch.cuda.synchronize()
  ```

## Phase 3: Real Robot Integration (1 week)
*Ensure GPU brain works with physical hardware*

### Latency-Optimized Pipeline
- [ ] **Zero-copy sensor integration** (Priority: CRITICAL)
  ```python
  class RobotGPUBridge:
      def __init__(self):
          # Pre-allocate pinned memory for sensor data
          self.sensor_buffer = torch.zeros(
              64, device='cuda', 
              pin_memory=True
          )
          
      def process_sensors_async(self, raw_sensors):
          # Direct copy to GPU without intermediate buffers
          self.sensor_buffer.copy_(raw_sensors, non_blocking=True)
          return self.sensor_buffer
  ```

### Adaptive Compute Budget
- [ ] **Dynamic quality scaling** (Priority: HIGH)
  ```python
  class AdaptiveGPUBrain:
      def adjust_quality(self, timing_ms):
          if timing_ms > 1.0:  # Over budget
              self.reduce_quality()
              # - Reduce pattern candidates
              # - Skip non-critical updates
              # - Use lower precision
          elif timing_ms < 0.5:  # Under budget
              self.increase_quality()
              # - More pattern discovery
              # - Higher precision operations
  ```

### Fallback Mechanisms
- [ ] **CPU fallback for edge cases** (Priority: MEDIUM)
  - Graceful degradation if GPU unavailable
  - Hybrid CPU/GPU for development
  - Emergency reflexes on CPU

## Phase 4: Performance Validation (1 week)
*Verify we achieved our targets*

### Benchmark Suite
- [ ] **Automated performance tests** (Priority: HIGH)
  ```python
  class GPUBenchmark:
      def run_all(self):
          results = {
              'field_evolution': self.benchmark_evolution(),
              'gradient_extraction': self.benchmark_gradients(),
              'pattern_discovery': self.benchmark_patterns(),
              'full_cycle': self.benchmark_full_cycle(),
              'memory_bandwidth': self.measure_bandwidth(),
              'kernel_efficiency': self.measure_kernel_efficiency()
          }
          assert results['full_cycle'] < 1.0  # Must be under 1ms
          return results
  ```

### Real Robot Testing
- [ ] **Deploy to PiCar-X** (Priority: CRITICAL)
  - Measure end-to-end latency
  - Verify behavioral preservation
  - Test under real-world conditions
  - **Success criteria**:
    - Response time <10ms (sensor to motor)
    - Smooth behavioral trajectories
    - No observable stuttering

### Cognitive Validation
- [ ] **Verify intelligence preservation** (Priority: CRITICAL)
  - Run behavioral test suite
  - Compare emergence metrics
  - Ensure no loss of cognitive capability
  - **Required tests**:
    ```bash
    python server/tools/testing/behavioral_test_fast.py
    python demos/emergent_navigation_demo.py
    python tests/test_field_integration.py
    ```

## Performance Targets

### Hard Requirements
- **Cycle time**: <1ms on RTX 4090, <5ms on Jetson Orin
- **Memory usage**: <4GB GPU RAM
- **Allocation rate**: 0 allocations/cycle during steady state
- **CPU usage**: <5% during normal operation

### Optimization Targets
- **Field evolution**: <0.1ms
- **Gradient extraction**: <0.05ms
- **Pattern matching**: <0.1ms
- **Motor generation**: <0.05ms
- **Total overhead**: <0.2ms

## Code Changes Required

### Priority 1: Core Files
1. `server/src/brains/field/unified_field_brain.py`
   - Remove all NumPy operations
   - Pre-allocate all tensors
   - Fuse evolution operations

2. `server/src/brains/field/evolved_field_dynamics.py`
   - Convert to pure GPU operations
   - Implement fused evolution kernel
   - Optimize memory layout

3. `server/src/brains/field/motor_cortex.py`
   - GPU-based gradient to motor mapping
   - Remove CPU conversions
   - Vectorize all operations

### Priority 2: Supporting Systems
4. `server/src/brains/field/predictive_field_system.py`
   - GPU-based prediction
   - Batched error computation

5. `server/src/brains/field/unified_pattern_system.py`
   - GPU pattern library
   - Parallel pattern matching

6. `server/src/brains/field/field_strategic_planner.py`
   - GPU pattern discovery
   - Parallel candidate evaluation

### Priority 3: Infrastructure
7. `server/src/config/gpu_memory_manager.py`
   - Implement unified memory pool
   - Add profiling hooks
   - Memory bandwidth monitoring

8. `server/src/core/brain_telemetry.py`
   - Add GPU metrics
   - Kernel timing analysis
   - Memory usage tracking

## Risk Mitigation

### Technical Risks
- **Risk**: Custom CUDA kernels too complex
  - **Mitigation**: Start with PyTorch, optimize incrementally
  
- **Risk**: Memory bandwidth bottleneck
  - **Mitigation**: Optimize tensor layout, use texture memory

- **Risk**: Loss of emergent behaviors
  - **Mitigation**: Extensive behavioral testing at each step

### Timeline Risks
- **Risk**: Unforeseen GPU compatibility issues
  - **Mitigation**: Develop on multiple GPU types early

- **Risk**: Real robot integration delays
  - **Mitigation**: Use simulation for initial validation

## Success Criteria

### Week 1 Checkpoint
- [ ] All NumPy operations removed
- [ ] Memory pre-allocation complete
- [ ] Basic GPU profiling working
- [ ] 10x speedup achieved

### Week 2 Checkpoint  
- [ ] Fused evolution kernel implemented
- [ ] Pattern operations on GPU
- [ ] 50x speedup achieved

### Week 3 Checkpoint
- [ ] Custom kernels for bottlenecks (if needed)
- [ ] Memory bandwidth optimized
- [ ] 100x speedup achieved

### Week 4 Checkpoint
- [ ] Robot integration tested
- [ ] Adaptive compute working
- [ ] <1ms cycle time verified

### Final Validation
- [ ] Behavioral tests pass
- [ ] Real robot deployment successful
- [ ] Performance targets met
- [ ] Code ready for production

## Next Steps

1. **Immediate**: Start with memory pre-allocation (biggest quick win)
2. **This week**: Remove all NumPy operations from hot path
3. **Next week**: Implement fused evolution kernel
4. **Following week**: Real robot testing

---

*"Make it work, make it right, make it fast - in that order. We've done the first two, now let's make it fly."*