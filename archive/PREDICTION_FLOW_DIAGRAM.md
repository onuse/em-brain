# Prediction System Flow Diagram

## Prediction Generation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SimplifiedUnifiedBrain                            │
│                                                                      │
│  1. process_robot_cycle() receives sensory input                    │
│  2. Stores in recent_sensory deque (last 10 values)                │
│  3. Calls _generate_prediction() if phase 1 enabled                 │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PredictiveFieldSystem                             │
│                                                                      │
│  generate_sensory_prediction():                                      │
│  1. Initialize predictions & confidences to zero                     │
│  2. If Phase 3 enabled: Add hierarchical predictions (0.4 weight)   │
│  3. Momentum prediction from recent_sensory (0.3 weight)            │
│  4. Topology region predictions (variable weight)                    │
│  5. Field activity bias (0.1 weight)                                │
│  6. Clamp to [-1, 1] range                                          │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
                ▼                                           ▼
┌──────────────────────────────┐         ┌─────────────────────────────┐
│   Momentum Prediction        │         │   Topology Regions          │
│                              │         │                             │
│  - Linear extrapolation      │         │  For each region:           │
│  - Uses last 2-3 values      │         │  - Extract local temporal   │
│  - Weight: 0.3               │         │  - Map to sensor indices    │
│  - Base confidence: 0.2      │         │  - Weight by confidence     │
└──────────────────────────────┘         └─────────────────────────────┘
```

## Confidence Calculation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Actual vs Predicted Comparison                    │
│                                                                      │
│  1. Calculate error = abs(actual - predicted) per sensor            │
│  2. Average error across all sensors                                 │
│  3. Confidence = 1.0 - min(1.0, error * 2.0)                       │
│  4. Smooth confidence with history (0.95 old + 0.05 new)           │
└─────────────────────────────────────────────────────────────────────┘
```

## Learning Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PredictionErrorLearning (Phase 2)                 │
│                                                                      │
│  1. Receives tensor prediction errors                                │
│  2. Maps errors to spatial field representation                      │
│  3. High error → Strong learning signal                              │
│  4. Low error → Consolidation signal                                 │
│  5. Modulates field dynamics based on errors                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Region Development Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TopologyRegionSystem                              │
│                                                                      │
│  1. detect_topology_regions() finds stable patterns                 │
│  2. assign_sensory_predictions() assigns sensors to regions         │
│  3. Regions with poor prediction (<0.2) reassigned                  │
│  4. Regions with good prediction (>0.7) can take more sensors       │
│  5. update_prediction_success() updates confidence                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Issues Identified

### 1. Bootstrap Problem
- Initial predictions are essentially random
- Random predictions → high error → 0% confidence
- 0% confidence → high exploration → unstable patterns
- Unstable patterns → poor region formation → poor predictions

### 2. Weight Distribution
- Momentum: 30% (works immediately but limited)
- Hierarchical: 40% (Phase 3 - currently disabled)
- Regions: Variable (starts at 0, needs time to develop)
- Field bias: 10% (minimal contribution)

### 3. Region Assignment
- Regions assigned sensors in round-robin fashion
- No initial affinity between regions and sensors
- Takes time for regions to specialize

### 4. Confidence Formula
- `confidence = 1.0 - min(1.0, error * 2.0)`
- Any error > 0.5 results in 0% confidence
- Very harsh for initial learning

## Recommendations

### Quick Fixes
1. **Soften confidence formula**: Use `1.0 - min(1.0, error)` or add floor
2. **Increase momentum weight**: Try 0.5 instead of 0.3
3. **Add prediction damping**: Blend with neutral value (0.5) initially

### Structural Improvements
1. **Initialize regions with sensory bias**: Use sensory history to guide initial assignment
2. **Add prediction priors**: Simple heuristics like "sensors change slowly"
3. **Implement prediction smoothing**: Temporal low-pass filter on predictions

### Long-term Evolution
1. **Enable Phase 3**: Hierarchical predictions could provide stability
2. **Add meta-learning**: Learn learning rates from prediction success
3. **Implement curiosity-driven exploration**: Target high-uncertainty regions