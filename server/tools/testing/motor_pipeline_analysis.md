# Motor Pipeline Analysis & Improvement Plan

## Issues Identified

### 1. **Motor Command Mapping Mismatch**
- **Problem**: Pattern motor generates `[forward-lateral, turn, speed, action]`
- **Test expects**: `[forward, left, right, stop]`
- **Result**: Motor[0] = `forward - lateral`, which could be negative or zero if lateral is high

### 2. **Confidence Scaling Too Aggressive**
```python
commands = commands * (0.5 + 0.5 * tendencies['confidence'])
```
- With low confidence (0.2), commands are scaled by 0.6
- This makes weak movements even weaker
- Compounds with motor cortex amplification

### 3. **Exploration Score Calculation**
- Grid-based: 20x20 = 400 cells
- Score 0.003 = visiting only 1.2 cells
- Robot is essentially stuck in place

### 4. **Energy System Fighting Movement**
- Energy dropping to 0.03 (below 0.1 target)
- Low energy → reduced spontaneous dynamics
- Reduced spontaneous → less exploration

## Root Cause Analysis

The robot isn't moving because:
1. Motor commands are too weak after confidence scaling
2. Motor mapping doesn't align with test expectations
3. Low energy reduces spontaneous exploration
4. Feedback loop: no movement → no novelty → no exploration → no movement

## Holistic Improvement Plan

### Phase 1: Fix Motor Mapping (Immediate)
```python
# Option A: Change pattern motor output to match test expectations
if self.motor_dim >= 4:
    # Map to [forward, left, right, stop] for compatibility
    commands[0] = tendencies['forward']  # Pure forward
    commands[1] = max(0, tendencies['turn'])  # Left turn
    commands[2] = max(0, -tendencies['turn'])  # Right turn  
    commands[3] = tendencies['stop'] if 'stop' in tendencies else 0.0

# Option B: Fix test interpretation
# Change biological_embodied_learning to interpret motors correctly
```

### Phase 2: Adjust Confidence Scaling
```python
# Less aggressive scaling - maintain minimum movement
MIN_COMMAND_SCALE = 0.7  # Never scale below 70%
confidence_scale = MIN_COMMAND_SCALE + (1.0 - MIN_COMMAND_SCALE) * tendencies['confidence']
commands = commands * confidence_scale
```

### Phase 3: Boost Motor Cortex for Low Confidence
```python
# Current: Only boosts if confidence < 0.05
# Proposed: More aggressive assistance
if confidence < 0.3:  # Help more often
    # Ensure some forward movement for exploration
    boost_factor = 1.5 + (0.3 - confidence) * 2.0  # Up to 2.1x at 0 confidence
```

### Phase 4: Energy System Refinement
- Set minimum energy floor at 0.2 (not 0.1)
- Make decay stop at 0.2 
- Stronger energy injection when below 0.5
- Decouple exploration from energy (use baseline exploration)

### Phase 5: Exploration Baseline
```python
# Add minimum exploration regardless of other factors
EXPLORATION_BASELINE = 0.2  # Always have 20% exploration drive
total_exploration = EXPLORATION_BASELINE + (
    0.4 * spontaneous_exploration +
    0.3 * pattern_exploration +
    0.1 * learning_exploration
)
```

## Recommended Parameter Changes

### Motor Cortex
- `activation_threshold`: 0.1 → 0.05 (accept weaker signals)
- `confidence_threshold`: 0.05 → 0.1 (help low confidence more)
- `max_amplification`: 3.0 → 4.0 (stronger boost)

### Pattern Motor
- Remove or reduce confidence scaling
- Add exploration baseline
- Fix motor mapping

### Energy System
- `target_min_energy`: 0.1 → 0.2
- `field_decay_rate`: 0.999 → 0.9995 (slower decay)
- Decay stops at `target_min_energy`

## Testing Strategy

1. **Unit Test Motor Pipeline**
   - Feed known inputs, verify outputs
   - Test each component separately
   
2. **Integration Test**
   - Track motor commands through full pipeline
   - Verify exploration increases
   
3. **Movement Validation**
   - Simple grid world
   - Measure actual coverage
   - Target: >10% grid coverage (40+ cells)

## Implementation Priority

1. **Fix motor mapping** (Critical - robots can't move properly)
2. **Adjust confidence scaling** (High - improves movement)
3. **Energy system refinement** (Medium - stability)
4. **Exploration baseline** (Medium - ensures movement)
5. **Parameter tuning** (Low - optimization)