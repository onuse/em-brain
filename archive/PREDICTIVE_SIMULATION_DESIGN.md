# Predictive Simulation Design Proposal

## Current Architecture Analysis

### Existing Prediction Flow
1. **Field Evolution** → Predicted field state (simple decay model)
2. **Sensory Prediction** → PredictiveFieldSystem generates expected sensors
3. **Action Generation** → ActionPredictionSystem creates candidates with predicted outcomes
4. **Action Selection** → Choose based on predictions + exploration drive

### Key Observation
The architecture already has **action candidates with predicted outcomes**! But currently these predictions are simple linear projections. The missing piece is **testing these predictions through simulation**.

## Natural Integration Points

### 1. Enhancement of Action Candidates (Most Natural)
```python
# Current: ActionPredictionSystem generates candidates with simple predictions
candidates = self.action_prediction.generate_action_candidates(
    current_predictions=current_predictions,
    n_candidates=5
)

# Enhanced: Each candidate gets GPU-simulated futures
simulated_candidates = self.future_simulator.evaluate_candidates(
    candidates=candidates,
    current_field=self.unified_field,
    horizon=20  # Simulate 20 cycles ahead
)
```

### 2. Parallel to Field Evolution (Minimal Change)
```python
def _evolve_field(self):
    """Evolve field dynamics."""
    # Current evolution (CPU)
    self.unified_field = self.field_dynamics.evolve_field(self.unified_field)
    
    # NEW: Parallel future simulation (GPU)
    if self.future_simulator is not None:
        # Fork current state and simulate futures asynchronously
        self.future_states = self.future_simulator.simulate_async(
            self.unified_field, 
            n_futures=32
        )
```

### 3. Inside Action Selection (Most Integrated)
```python
def select_action(self, candidates, hierarchical_predictions, exploration_drive):
    # Current: Score based on static predictions
    scores = self._score_static_predictions(candidates)
    
    # Enhanced: Score based on simulated outcomes
    if self.gpu_futures_ready():
        simulated_scores = self._score_simulated_futures(
            candidates, 
            self.future_states
        )
        # Blend static and simulated scores
        scores = 0.3 * scores + 0.7 * simulated_scores
```

## Proposed Design: GPU Future Simulator

### Core Architecture
```python
class GPUFutureSimulator:
    """
    Simulates multiple possible futures in parallel on GPU.
    Designed to integrate seamlessly with existing prediction systems.
    """
    
    def __init__(self, field_shape, n_futures=32, horizon=20, device='mps'):
        self.n_futures = n_futures
        self.horizon = horizon
        self.device = device
        
        # Simplified field dynamics for GPU (no Python loops)
        self.gpu_field_dynamics = self._create_gpu_dynamics()
        
        # Future states buffer (kept on GPU)
        self.future_buffer = torch.zeros(
            n_futures, horizon, *field_shape, device=device
        )
        
    def evaluate_action_candidates(self, 
                                 candidates: List[PredictiveAction],
                                 current_field: torch.Tensor,
                                 confidence: float) -> List[SimulatedAction]:
        """
        For each action candidate, simulate multiple futures.
        Returns enhanced candidates with simulation results.
        """
        simulated_actions = []
        
        for candidate in candidates:
            # Fork futures with this action
            futures = self._fork_futures_with_action(
                current_field, 
                candidate.motor_values,
                n_variations=8  # 8 futures per action
            )
            
            # Simulate forward
            outcomes = self._simulate_batch(futures, self.horizon)
            
            # Analyze outcomes
            analysis = self._analyze_futures(outcomes)
            
            # Enhance candidate with simulation data
            simulated_actions.append(SimulatedAction(
                original=candidate,
                outcome_variance=analysis['variance'],
                outcome_stability=analysis['stability'],
                surprise_potential=analysis['surprise'],
                convergence_time=analysis['convergence']
            ))
            
        return simulated_actions
```

### Key Design Principles

1. **Non-Invasive Integration**
   - Works alongside existing systems, doesn't replace them
   - Falls back gracefully if GPU unavailable
   - Enhances rather than replaces current predictions

2. **Biologically Inspired**
   - Multiple futures = neural population coding
   - Variance across futures = uncertainty
   - Convergence = confidence in prediction

3. **GPU-Efficient Implementation**
   ```python
   def _simulate_batch(self, initial_states, horizon):
       """All futures simulated in single batch operation"""
       states = initial_states
       trajectory = []
       
       for t in range(horizon):
           # Single batched evolution for all futures
           states = self.gpu_field_dynamics(states)
           trajectory.append(states)
           
       return torch.stack(trajectory, dim=1)
   ```

## Integration Flow

### Phase 1: Observation Mode
- Future simulator runs in parallel but doesn't influence decisions
- Logs prediction accuracy vs simulated outcomes
- Builds confidence in simulation quality

### Phase 2: Advisory Mode  
- Simulation results shown as "confidence adjustment"
- High variance in futures → reduce action confidence
- Convergent futures → increase action confidence

### Phase 3: Full Integration
- Action scores directly incorporate simulation results
- Learn to trust simulations based on their accuracy
- Dynamic weighting between static and simulated predictions

## Why This Design Feels Natural

1. **Enhances Existing Flow** - The action candidates already have predictions; we're just making them better
2. **Preserves Biological Realism** - The core brain still works the same way
3. **Graceful Degradation** - Works without GPU, just less rich
4. **Learning Integration** - Prediction errors from simulations feed back into learning

## Implementation Touchpoints

### Minimal Changes Required:
1. Add `future_simulator` to SimplifiedUnifiedBrain.__init__
2. Call `evaluate_candidates()` in `_generate_motor_action()` 
3. Add simulation scores to action selection
4. Optional: Show futures in brain state for monitoring

### No Changes To:
- Field dynamics
- Sensory prediction
- Topology regions  
- Pattern systems
- Core brain cycle

## Performance Considerations

- **CPU Load**: Unchanged (simulation on GPU)
- **GPU Load**: +40-60% (good use of idle capacity)
- **Memory**: +~100MB for future buffers
- **Latency**: Can hide behind CPU operations

## Next Steps

1. **Prototype** the GPUFutureSimulator class
2. **Test** with simple integration in observation mode
3. **Measure** impact on decision quality
4. **Iterate** on future forking strategies

This design maintains the biological elegance while adding genuine predictive power through parallel simulation.