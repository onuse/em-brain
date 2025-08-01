# Minimal Confidence Dynamics Proposal

## Core Insight

The Dunning-Kruger effect naturally emerges from two simple facts:
1. Simple models can't detect their own incompleteness
2. Confidence should reflect both prediction success AND model uncertainty

## Minimal Implementation

Instead of a separate confidence dynamics system, we can achieve natural confidence evolution with a few small changes:

### 1. Confidence Formula Based on Model State

```python
# Current harsh formula
confidence = 1.0 - min(1.0, error * 2.0)

# Natural formula that considers model development
model_uncertainty = 1.0 / (1.0 + len(topology_regions) / 10.0)
error_weight = 1.0 + model_uncertainty  # 2.0 when new, 1.0 when developed

confidence = 1.0 - min(1.0, error * error_weight)
```

This naturally gives:
- Early (few regions): High error_weight → More forgiving → Higher confidence
- Later (many regions): Low error_weight → Less forgiving → Earned confidence

### 2. Prediction Confidence Momentum

```python
# Add simple momentum to confidence
confidence_momentum = 0.9 - (brain_cycles / 1000.0) * 0.2  # Decreases from 0.9 to 0.7
smoothed_confidence = confidence_momentum * previous + (1 - momentum) * current
```

This gives:
- Early: High momentum → Slow to lose confidence (optimistic)
- Later: Lower momentum → Faster adaptation (realistic)

### 3. Base Confidence from Model Simplicity

```python
# When no predictions yet, confidence reflects model simplicity
if not has_predictions:
    # Simple model = higher base confidence (doesn't know what it doesn't know)
    base_confidence = 0.5 * (1.0 - model_complexity)
else:
    base_confidence = 0.1  # Low floor when predictions exist
```

## Why This Works

1. **No explicit stages** - Confidence naturally evolves with model complexity
2. **Minimal code** - Just 3-4 lines change in existing confidence calculation
3. **Theoretically sound** - Based on information theory (simple models can't see their limitations)
4. **Natural gradient** - Smooth transition from naive to calibrated

## Implementation Sketch

```python
# In process_robot_cycle, replace confidence calculation with:

# Model complexity (0 = simple, 1 = complex)
model_complexity = min(1.0, len(self.topology_region_system.regions) / 50.0)

# Error weight decreases as model develops
error_weight = 1.5 - 0.5 * model_complexity  # 1.5 → 1.0

# Base confidence higher for simple models
base_confidence = 0.2 * (1.0 - model_complexity) if self.brain_cycles < 50 else 0.0

# Calculate confidence with natural dynamics
raw_confidence = max(base_confidence, 1.0 - min(1.0, sensory_error * error_weight))

# Momentum decreases over time
momentum = 0.9 - min(0.2, self.brain_cycles / 1000.0)
self._current_prediction_confidence = (
    momentum * self._current_prediction_confidence + 
    (1.0 - momentum) * raw_confidence
)
```

## Benefits

1. **Invisible** - No new classes or systems
2. **Natural** - Emerges from model development
3. **Tunable** - Just 2-3 parameters
4. **Reversible** - Easy to remove if not helpful

## Result

Early brain: Simple model + forgiving error weight + high momentum = Natural optimism
Late brain: Complex model + strict error weight + low momentum = Calibrated confidence

The Dunning-Kruger effect emerges without being explicitly programmed.