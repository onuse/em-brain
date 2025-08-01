# Learning Dynamics Proposal: From Confidence to Coherence

## The Core Insight

We've been conflating two distinct concepts:
1. **Prediction accuracy** (how well we predict the external world)
2. **Model coherence** (how stable and self-consistent our internal model is)

"Confidence" was our attempt to bridge these, but it's the wrong abstraction. What the brain actually needs is to modulate learning based on **coherence**, not accuracy.

## The Natural Learning Dynamics

A learning system naturally goes through these phases:

1. **Incoherent** → Learn aggressively (high surprise, unstable model)
2. **Becoming coherent** → Learn selectively (model stabilizing)
3. **Coherent** → Learn minimally (stable model, low surprise)
4. **Disrupted** → Return to learning (coherence broken by novelty)

## Proposed Implementation

Replace "confidence" with three clean signals:

### 1. Model Coherence (Internal Signal)
```python
# How stable is the field over recent cycles?
field_stability = 1.0 - field_change_rate
pattern_stability = stable_patterns / total_patterns
temporal_coherence = 1.0 - temporal_variance

model_coherence = (field_stability + pattern_stability + temporal_coherence) / 3.0
```

This measures: "Is my internal model consistent with itself?"

### 2. Surprise Level (External Signal)
```python
# How different is current input from recent history?
if has_recent_history:
    input_deviation = distance(current_input, recent_average)
    surprise = sigmoid(input_deviation * sensitivity)
else:
    surprise = 0.5  # Neutral when no history
```

This measures: "Is the world showing me something unexpected?"

### 3. Learning Pressure (Derived Signal)
```python
# Combine internal and external signals
if model_coherence < 0.3:
    # Incoherent model: learn everything
    learning_pressure = 0.9
elif surprise > 0.7:
    # Coherent but surprised: targeted learning
    learning_pressure = 0.5 + 0.4 * surprise
else:
    # Coherent and unsurprised: minimal learning
    learning_pressure = 0.1 + 0.2 * (1.0 - model_coherence)
```

## Why This Is Better

1. **No prediction needed**: Works even when predictions are poor/absent
2. **Natural dynamics**: Coherence naturally emerges as patterns stabilize
3. **Honest signals**: We measure what we actually use (stability, surprise)
4. **Simpler**: No need to compute prediction errors for confidence

## The Beautiful Part

This gives us natural Dunning-Kruger for free:
- Early: Low coherence → High learning → Model quickly becomes coherent with limited data
- Middle: Disruptions break coherence → Reality is complex → Coherence drops
- Late: True coherence emerges → Stable despite surprises

## Code Changes

1. Replace `_current_prediction_confidence` with:
   - `_model_coherence`
   - `_surprise_level`
   - `_learning_pressure`

2. Modify imprinting:
```python
# Instead of surprise_factor = 1.0 - confidence
scaled_intensity *= self._learning_pressure
```

3. Modify exploration:
```python
# Instead of uncertainty_bonus = (1.0 - confidence) * 0.2
exploration_drive = base + (1.0 - self._model_coherence) * 0.3
```

## The Deep Insight

**Learning systems don't need to know if they're "right" - they need to know if they're "ready".**

A coherent model that's wrong will naturally discover its wrongness through surprise. An incoherent model that happens to be accurate will still benefit from becoming coherent. The system self-regulates without needing an external accuracy measure.

This feels more fundamental - it's about the system's relationship with itself, not its relationship with ground truth.

## Implementation Analysis

After thorough analysis of the codebase, here's what we found:

### Current Confidence Usage
The system uses confidence for:
1. **Learning rate modulation** - surprise-based imprinting strength
2. **Exploration/exploitation balance** - cognitive mode selection
3. **Action selection** - confidence-weighted scoring
4. **Spatial attention** - uncertainty map generation
5. **Prediction combination** - multi-timescale weighting

### Scope of Changes
Replacing confidence with coherence/surprise/pressure would require:
- **15+ files modified** across core systems
- **3D decision boundaries** instead of simple thresholds
- **New combination mathematics** for predictions
- **Complete recalibration** of all behavioral parameters
- **Persistence and telemetry overhaul**

### Honest Assessment

**The hard truth**: While the coherence/surprise/pressure model is theoretically cleaner, the implementation complexity is substantial. The current confidence system, despite its conceptual muddiness, is:
- Well-integrated throughout the codebase
- Computationally efficient (single scalar)
- Easy to tune and debug
- Already working effectively

**The verdict**: This is a case where pragmatism should win over theoretical purity. The current confidence system is a "good enough" abstraction that serves multiple purposes adequately.

## Alternative: Minimal Refinement

Instead of a full replacement, consider minimal improvements:

1. **Rename for clarity**: Call it `model_stability` instead of `prediction_confidence`
2. **Document the dual nature**: Explicitly state it measures both prediction accuracy AND model coherence
3. **Keep the simple formula**: The current natural D-K implementation is already elegant

Sometimes the best insight is knowing when not to change something.