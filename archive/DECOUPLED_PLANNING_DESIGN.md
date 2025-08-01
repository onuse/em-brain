# Decoupled Planning Design

## The Core Insight

Current architecture (blocking):
```
Sensory → Generate Actions → Simulate All Futures (8s) → Select → Motor
```

Proposed architecture (non-blocking):
```
Sensory → Use Cached Plan/React → Motor (fast path: 2s)
     ↓
Background Planning Thread (slow path: continuous)
```

## Biological Basis

This mirrors how brains actually work:
- **Fast path**: Reflexes, habits, cached plans (100-500ms)
- **Slow path**: Deliberation, planning, simulation (1-10s)

You don't re-plan every muscle movement - you execute cached plans and update them asynchronously!

## Implementation Design

### 1. Dual-Mode Action Selection
```python
class DecoupledActionSystem:
    def __init__(self):
        # Fast path: simple predictions or cached plans
        self.cached_plan = None
        self.plan_confidence = 0.0
        
        # Slow path: rich simulation
        self.planning_thread = None
        self.future_plans = Queue()
        
    def get_action(self, current_state, timeout=0.1):
        """Get action with timeout - use best available option"""
        
        # Check if we have a fresh simulated plan
        try:
            new_plan = self.future_plans.get_nowait()
            self.cached_plan = new_plan
            self.plan_confidence = new_plan.confidence
        except Empty:
            pass
        
        # Decide based on confidence and staleness
        if self.should_use_cached_plan():
            # Fast path: execute cached plan
            return self.cached_plan.get_next_action()
        else:
            # Fallback: simple reactive action
            return self.generate_reactive_action(current_state)
```

### 2. Continuous Background Planning
```python
class BackgroundPlanner(Thread):
    """Runs continuously, always improving plans"""
    
    def run(self):
        while True:
            # Get latest brain state
            current_field = self.brain.unified_field.clone()
            
            # Simulate futures (takes 8s)
            future_plans = self.simulate_comprehensive_futures(
                current_field,
                n_futures=32,  # Can afford more since not blocking
                horizon=50     # Longer horizon
            )
            
            # Rank plans by quality
            best_plans = self.rank_plans(future_plans)
            
            # Queue for main thread
            self.output_queue.put(best_plans[0])
```

### 3. Plan Execution and Monitoring
```python
class PlanExecutor:
    """Executes multi-step plans with monitoring"""
    
    def __init__(self):
        self.current_plan = None
        self.plan_step = 0
        self.plan_divergence = 0.0
        
    def execute_step(self, actual_state):
        if self.current_plan is None:
            return self.reactive_action(actual_state)
            
        # Get planned action for this step
        planned_action = self.current_plan.actions[self.plan_step]
        
        # Check if reality matches prediction
        expected_state = self.current_plan.expected_states[self.plan_step]
        self.plan_divergence = self.measure_divergence(actual_state, expected_state)
        
        if self.plan_divergence > threshold:
            # Plan invalid - request replan
            self.request_new_plan(urgent=True)
            return self.reactive_action(actual_state)
        else:
            # Plan still valid - execute
            self.plan_step += 1
            return planned_action
```

## Three-Layer Architecture

### Layer 1: Reflexive (10-100ms)
- Hard-coded safety responses
- Basic obstacle avoidance
- Emergency stops

### Layer 2: Habitual (100-2000ms)
- Cached plans from Layer 3
- Simple predictive actions
- Current implementation

### Layer 3: Deliberative (1-10s)
- GPU future simulation
- Complex multi-step plans
- Runs continuously in background

## Benefits

1. **No blocking** - Robot keeps moving while thinking
2. **Better plans** - Can simulate 100s of futures instead of 8
3. **Adaptive** - Smooth degradation from planned to reactive
4. **Biological** - Matches fast/slow thinking systems

## Integration Points

### Minimal Change Version
```python
# In SimplifiedUnifiedBrain.__init__
self.planning_executor = PlanExecutor()
self.background_planner = BackgroundPlanner(self) if use_planning else None

# In _generate_motor_action
if self.planning_executor.has_valid_plan():
    # Fast path - use cached plan
    return self.planning_executor.execute_step(self.unified_field)
else:
    # Slow path - current implementation
    return self._generate_reactive_action()
```

### Advanced Version
- Multiple planning threads at different timescales
- Hierarchical plans (strategic → tactical → immediate)
- Plan library that learns successful patterns

## Challenges

1. **Synchronization** - Plans become stale
2. **Reality divergence** - When to abandon plans
3. **Resource management** - GPU contention

## Why This Feels Right

This solves multiple problems:
- Performance (no more 8s blocking)
- Biological realism (fast/slow paths)
- Scalability (can add more planning threads)
- Intelligence (much richer plans possible)

It's not a hack - it's how intelligent systems actually work. You don't recalculate everything every moment; you follow plans and update them asynchronously.