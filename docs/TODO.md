# TODO: Field-Native Strategic Planning

## Major Pivot Summary

We've pivoted from low-level action simulation to field-native strategic patterns. Instead of simulating "if I move left 0.3..." we now discover field configurations that naturally create beneficial behaviors.

## Current State

### What We Built (Now Deprecated)
1. **GPUFutureSimulator** - Simulates exact future field states for action sequences
2. **CachedPlanSystem** - Tries to cache action plans (0% hit rate)
3. **Decoupled planning** - Still takes 500-1800ms for "reactive" actions
4. **SimulatedAction/PredictiveAction** - Explicit action representations

### Why It Failed
- Plans become stale in 100ms but take 6s to compute
- Context matching too fragile for real-world variation
- Not actually decoupled (reactive path still slow)
- Fighting the field-native philosophy with symbolic structures

## New Direction: Strategic Field Patterns

### Core Insight
"Strategy" is not something you have, it's something you ARE. Strategic planning means discovering field configurations that naturally bias behavior toward good outcomes.

### Implementation Plan

#### Phase 1: Field Pattern Discovery (COMPLETED)
Transform GPUFutureSimulator into FieldStrategicPlanner:
- [x] Identify memory channels for persistent patterns (32-47)
  - Use temporal feature channels that already have slower decay
  - Ensure pattern installation doesn't interfere with sensory integration
  - Add dedicated persistence factor (0.95-0.99) for strategic patterns
- [x] Create pattern generation methods:
  - **Gradient patterns**: Directional flow (approach/avoid behaviors)
  - **Radial patterns**: Centering behaviors (stay near/far from points)
  - **Wave patterns**: Oscillatory behaviors (patrol, search)
  - **Sparse activation**: Focused behaviors (attend to specific features)
  - **Learned combinations**: Blend successful patterns
- [x] Implement pattern evaluation through field evolution:
  ```python
  # Simulate 50-100 steps with pattern installed
  # Evaluate: stability, reward correlation, behavioral coherence
  # Score = trajectory_value + pattern_persistence + behavior_smoothness
  ```
- [x] Test pattern persistence and influence:
  - Verify patterns maintain 80%+ correlation after 30 cycles
  - Confirm patterns create consistent behavioral biases
  - Measure influence decay rate

#### Phase 2: Pattern Integration (COMPLETED)
Modify brain to use strategic patterns:
- [x] Add pattern installation in memory channels:
  ```python
  # In SimplifiedUnifiedBrain._generate_motor_action():
  # 1. Check if strategic pattern exists
  # 2. If yes, blend into channels 32-47 with strength 0.3-0.5
  # 3. Let normal field evolution create motor output
  # NO explicit "if pattern then action" logic!
  ```
- [x] Ensure patterns persist across cycles:
  - Modify field evolution to preserve channels 32-47 with 0.95+ persistence
  - Add pattern refresh mechanism (reapply at 0.1 strength each cycle)
  - Implement pattern fade-out when switching strategies
- [x] Remove explicit action planning code:
  - Delete `SimulatedAction` class and related structures
  - Remove action sequence generation from GPUFutureSimulator
  - Simplify `select_action()` to work with field gradients only
- [x] Test emergent behaviors from patterns:
  - Gradient pattern → forward movement emerges
  - Radial pattern → centering behavior emerges
  - Wave pattern → oscillatory motion emerges
  - No explicit motor commands!

#### Phase 3: Pattern Library (COMPLETED)
Learn and store successful patterns:
- [x] Create pattern similarity metrics in field space:
  ```python
  # Cosine similarity in flattened pattern space
  # BUT also consider behavioral similarity:
  # - Do patterns create similar motion trajectories?
  # - Do they respond similarly to obstacles?
  # Behavioral similarity > structural similarity
  ```
- [x] Store patterns with context embeddings:
  - Context = compressed field state when pattern was successful
  - Use channels 0-31 mean/variance as simple context
  - Store: (pattern, context, success_score, behavioral_tags)
- [x] Implement pattern retrieval and blending:
  ```python
  def retrieve_pattern(current_context):
      # Find patterns with similar contexts
      # Blend top 2-3 patterns weighted by similarity
      # Add 10% novel variation
      return blended_pattern
  ```
- [x] Test pattern reuse across situations:
  - Same pattern works in slightly different environments
  - Blended patterns create intermediate behaviors
  - Library grows but plateaus at ~20-30 core patterns

#### Phase 4: Biological Validation (2-3 days)
Ensure biological plausibility:
- [ ] Verify patterns create smooth behavioral trajectories
- [ ] Test robustness to sensory noise
- [ ] Validate energy efficiency
- [ ] Confirm emergence without engineering

## Code to Remove/Refactor

### Completely Remove
- `cached_plan_system.py` - Action caching doesn't work
- `SimulatedAction` class - Too symbolic
- Action sequence planning in GPUFutureSimulator

### Refactor
- `GPUFutureSimulator` → `FieldStrategicPlanner`
- `evaluate_action_candidates()` → `discover_strategic_patterns()`
- `CachedPlanSystem` → `PatternLibrary` (field patterns, not actions)

### Keep But Simplify
- Async execution infrastructure (still useful for background pattern discovery)
- Adaptive configuration (but for pattern complexity, not action simulation)
- Fast reactive path (but truly fast: <100ms)

## Success Metrics

### Old (Flawed) Metrics
- ❌ Cache hit rate (was 0%)
- ❌ Planning time <2s (was achieving this but meaningless)
- ❌ Number of futures simulated (wrong level of abstraction)

### New (Meaningful) Metrics
- ✓ Pattern persistence (how long patterns remain influential)
- ✓ Behavioral coherence (smooth, purposeful movement)
- ✓ True reactive speed (<100ms with pattern influence)
- ✓ Pattern reuse rate (successful pattern library hits)
- ✓ Emergence score (behaviors not explicitly programmed)

## Timeline
- Week 1: Pattern discovery and integration
- Week 2: Pattern library and biological validation
- Week 3: Testing and refinement

## Key Implementation Insights

### 1. Pattern Discovery Algorithm
```python
def discover_pattern(field, reward_signal):
    best_pattern = None
    best_score = -inf
    
    for _ in range(n_candidates):
        # Generate candidate pattern
        pattern = generate_candidate()  # gradient, radial, wave, or sparse
        
        # Install in test field
        test_field = field.clone()
        test_field[:,:,:,32:48] = pattern
        
        # Simulate future evolution
        trajectory = simulate_evolution(test_field, steps=100)
        
        # Score based on:
        # - Reward correlation
        # - Behavioral coherence
        # - Pattern stability
        score = evaluate_trajectory(trajectory, reward_signal)
        
        if score > best_score:
            best_pattern = pattern
            best_score = score
            
    return best_pattern
```

### 2. Critical Success Factors
- **No explicit motor commands** - patterns shape field, field creates motion
- **Long time horizons** - patterns influence behavior for 30+ cycles
- **True emergence** - behaviors arise from field dynamics, not rules
- **Robustness** - patterns create tendencies, not rigid behaviors

### 3. What Makes This Different
- We're not planning actions, we're discovering beneficial field configurations
- We're not executing plans, we're letting patterns shape dynamics
- We're not caching decisions, we're building a library of behavioral attractors

## Critical Principle
Every feature must be field-native. No symbolic abstractions, no engineered goals, no explicit plans. Let patterns shape the flow, and behavior emerges.

---

*"The best strategy is not a plan, but a shape that guides the flow."*