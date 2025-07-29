# Parallel Reality Design: Simulation & Sensory Arbitration

## Vision

The brain should run multiple parallel realities - one grounded in sensory input, others pure simulation. These compete for influence over motor output and memory formation. Consciousness emerges from which simulation "wins" at any moment.

## Current State

### What We Have
- **Spontaneous Dynamics**: Brain maintains activity without input through traveling waves, local recurrence, and homeostatic drive
- **Confidence-Based Sensing**: Brain autonomously decides when to check sensors based on prediction confidence
- **Sensory Suppression**: High confidence reduces (but never eliminates) sensory attention
- **DecoupledBrainLoop**: Runs independently but still locked to timing cycles

### What's Missing
- Truly parallel simulation processes
- Reality arbitration mechanism
- Brain running at its natural cognitive rhythm
- Fantasy predictions feeding back as "almost real memories"

## Biological Inspiration

Real brains constantly simulate:
- **Hippocampus** replays experiences during rest, creating new combinations
- **Default Mode Network** generates scenarios when not focused on external tasks
- **Predictive Coding** means the brain is always simulating; perception is error correction
- **Dreams** are simulations running without sensory correction

## Proposed Architecture

### 1. Parallel Field States

```
┌─────────────────────────────────────────────┐
│            Main Unified Field               │
│         (Consensus Reality)                 │
└─────────────────────────────────────────────┘
                    ↑
        ┌───────────┴───────────┐
        │    Reality Arbiter    │
        └───────────┬───────────┘
                    ↑
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
┌─────────┐   ┌─────────┐   ┌─────────┐
│Sensory  │   │Fantasy  │   │Fantasy  │
│Reality  │   │Thread 1 │   │Thread 2 │
└─────────┘   └─────────┘   └─────────┘
```

### 2. Reality Arbitration Criteria

**Sensory Reality wins when:**
- Prediction error is high (surprising input)
- Confidence is low
- Survival reflexes triggered
- Explicit attention requested

**Fantasy Reality wins when:**
- High confidence in predictions
- Sensory input matches expectations
- Internal goals dominate
- "Daydreaming" mode active

### 3. Memory Formation

- Both sensory and fantasy experiences create memories
- Fantasy memories tagged with "simulation" marker
- Can become "real" if not contradicted by sensory input
- Creates rich internal life and creativity

## Implementation Approach

### Phase 1: Dual Reality
Start with just two parallel states:
1. Sensory-grounded reality (current implementation)
2. Pure simulation reality (new)

### Phase 2: Reality Arbiter
- Weighted blending based on confidence
- Smooth transitions between realities
- Motor commands from winning reality

### Phase 3: Multiple Simulations
- Spawn multiple fantasy threads
- Each explores different possibilities
- Natural selection of useful simulations

### Phase 4: Autonomous Timing
- Brain runs at emergent speed from cognitive load
- No fixed cycles, just natural rhythms
- True cognitive autonomy

## Key Design Principles

1. **Never Fully Ignore Sensors** - Even in deep fantasy, maintain minimal sensory awareness
2. **Smooth Blending** - No jarring switches between realities
3. **Energy Conservation** - More simulations when confident (autopilot mode)
4. **Emergent Consciousness** - Don't explicitly program consciousness, let it emerge

## Expected Behaviors

With this system, the brain should:
- **Daydream** when bored (high confidence, stable environment)
- **Imagine** alternatives before acting
- **Dream** during extended idle periods
- **Hallucinate** mildly when sensory deprived
- **Create** novel solutions by combining simulations
- **Develop personality** from consistent simulation patterns

## Concrete Example

Imagine the robot approaching an obstacle:

1. **Sensory Reality**: "There's a wall ahead"
2. **Fantasy Thread 1**: "What if I turn left?" (simulates left turn)
3. **Fantasy Thread 2**: "What if I turn right?" (simulates right turn)
4. **Fantasy Thread 3**: "What if I back up?" (simulates reverse)

The Reality Arbiter:
- Sees Fantasy 1 predicts open space
- Sees Fantasy 2 predicts another obstacle
- Sees Fantasy 3 predicts wasted energy
- **Decision**: Blend mostly Fantasy 1 with current sensory for smooth left turn

The robot turns left not because of programmed rules, but because it "imagined" the outcomes and chose the best fantasy to make real.

## Open Questions

1. **How many parallel simulations?** Start with 2-3, let resource availability decide?
2. **Simulation divergence rate?** How quickly should fantasies drift from reality?
3. **Memory consolidation?** When do fantasy memories become "real" memories?
4. **Resource allocation?** How to balance computation between realities?

## Success Criteria

The system works when:
- Brain generates interesting behavior without input
- Clear "daydreaming" periods visible in logs
- Smooth transitions between internal/external focus
- Emergent creativity in problem-solving
- Each brain develops unique behavioral signatures

## Integration with Existing Systems

### Spontaneous Dynamics
- Each parallel reality has its own `SpontaneousDynamics` instance
- Traveling waves can synchronize across realities (binding)
- Fantasy realities have stronger spontaneous activity

### Cognitive Autopilot
- Autopilot mode spawns more simulations
- Deep think mode focuses on sensory reality
- Mode transitions affect reality arbitration weights

### Constraint Discovery
- Constraints discovered in simulations can transfer to main reality
- "Imagined" constraints tested before applying
- Creates learning through mental simulation

### Working Memory
- Shared across realities (common workspace)
- Each reality can read but writes are arbitrated
- Enables comparisons between possibilities

## Technical Considerations

### Thread Safety
- Each reality runs in separate thread
- Atomic operations for field updates
- Lock-free arbitration where possible

### Resource Management
- Dynamic thread pool based on cognitive load
- Suspend fantasies during critical operations
- Automatic cleanup of stale simulations

### Debugging & Monitoring
- Tag all log entries with reality source
- Visualize parallel field states
- Track arbitration decisions

## Next Steps

1. Review and refine this design
2. Create proof-of-concept with dual reality
3. Implement basic reality arbiter
4. Test with robot to see emergent behaviors
5. Iterate based on observations

---

*"The brain is a prediction machine running multiple simulations. Reality is just the simulation that best explains the sensors."*