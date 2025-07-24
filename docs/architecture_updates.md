# Architecture Updates

## Changes to Add to ARCHITECTURE.md

### 1. Update Communication Architecture (line 222)

Replace:
```
# 24D sensor input -> 37D field processing -> 4D motor output
```

With:
```
# 25D sensor input -> 37D field processing -> 4D motor output
# Input: 24D sensors + 1D reward signal
```

### 2. Add Prediction System Section (after Learning System, around line 178)

```markdown
#### Prediction-Based Confidence
```python
# Field evolution serves as prediction mechanism
predicted_field = field.evolve()
actual_field = field.after_sensory_input()
prediction_error = compare(predicted, actual)
confidence = 1.0 / (1.0 + prediction_error * 5000)
```
- Field evolution naturally predicts next state
- Prediction errors drive learning
- Confidence based on actual prediction accuracy
- Creates genuine curiosity and surprise detection
```

### 3. Add Reward System Section (after Prediction System)

```markdown
#### Value Learning System
```python
# 25th sensory dimension carries reward signal
reward = sensory_input[24]  # -1.0 to +1.0
field.map_reward_to_energy_dimensions(reward)
field.strengthen_memories_by_importance(reward)
```
- External rewards shape field dynamics
- Positive rewards create stronger memories
- Negative rewards create aversive patterns
- Value gradients emerge in field topology
```

### 4. Update Emergent Properties (line 253)

Add:
```
- Prediction: Field evolution as future state anticipation
- Value: Reward-modulated field topology
- Curiosity: Seeking prediction errors for learning
```

### 5. Update Current Limitations (line 301)

Replace current limitations with:
```
### Current Limitations
- Constraint enforcement disabled (dimension indexing incompatibility)
- Behavioral differentiation weak (gradient-to-action mapping needs tuning)
- GPU processing limited to CPU due to MPS tensor dimension constraints
```

### 6. Add Implementation Details subsection on Memory Formation

```markdown
### Memory Formation
- **Topology Region Discovery**: Activation > 0.02, variance < 0.5
- **Region Persistence**: Removal only when activation < 0.001
- **Baseline Field Value**: 0.0001 (prevents zero without interfering)
- **Reward Modulation**: Positive rewards increase field intensity 0.5-1.0
```

## Summary

These updates document:
1. The prediction improvement addiction system
2. The reward/value learning system
3. Improved memory formation parameters
4. Current system capabilities and limitations

The architecture now reflects that the brain has:
- Real prediction-based confidence
- External value learning
- Persistent memory formation
- Intrinsic curiosity drive