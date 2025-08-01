# Strategic Planning Pivot: From Actions to Roadmaps

## The Insight

Instead of simulating 32 exact future field states to pick one action, use the GPU to create **abstract strategic plans** that guide behavior for many cycles.

## Current (Flawed) Approach
```
GPU: "If I move left 0.3, then forward 0.2, then right 0.1... 
      the field will be exactly [2M numbers]"
Brain: "Cool, I'll move left 0.3"
[World changes, plan is now useless]
```

## Pivoted (Strategic) Approach
```
GPU: "Strategic plan: 
      1. Navigate toward bright region
      2. Maintain safe distance from walls
      3. Explore dark areas when safe"
Brain: "I'll follow this strategy reactively"
[World changes, strategy still valid!]
```

## Architecture Transformation

### From: Field State Predictor
```python
class GPUFutureSimulator:
    def simulate_exact_futures(self, field, action):
        # Predicts exact field states
        # Useless after 100ms
```

### To: Strategic Planner
```python
class GPUStrategicPlanner:
    def create_strategy(self, field, goals):
        # Returns abstract plan:
        # - Behavioral primitives
        # - Goal sequences  
        # - Value gradients
        # - Constraint regions
        # Valid for seconds/minutes!
```

## What Strategic Plans Look Like

### 1. **Behavioral Waypoints**
```python
@dataclass
class StrategicPlan:
    waypoints: List[AbstractGoal]  # Not positions, but objectives
    constraints: List[SafetyRule]   # Always avoid these
    value_map: torch.Tensor        # Where good outcomes lie
    behavioral_mode: str           # "explore" / "exploit" / "escape"
    validity_duration: float       # How long this plan makes sense
```

### 2. **Abstract Goals, Not Actions**
Instead of: `[move_left: 0.3, move_forward: 0.2]`

We get: 
```python
goals = [
    "increase_brightness_signal",
    "maintain_wall_distance > 0.5",
    "visit_unexplored_regions",
    "return_to_safe_zone_if_threatened"
]
```

### 3. **Reactive Execution**
The fast brain interprets these goals reactively:
```python
def execute_strategy(self, current_state, strategic_plan):
    # Check which goal is active
    active_goal = strategic_plan.get_current_goal(current_state)
    
    # Generate reactive action toward goal
    if active_goal == "increase_brightness":
        return self.move_toward_brightness(current_state)
    elif active_goal == "maintain_wall_distance":
        return self.avoid_walls(current_state)
    # etc...
```

## Why This Works

### 1. **Plans Stay Valid**
- "Navigate toward light" works even if obstacles appear
- "Explore boundaries" remains meaningful as world changes
- Abstract goals are robust to environmental variation

### 2. **True Decoupling**
- GPU can spend 10 seconds creating a brilliant strategy
- Reactive brain executes at 50ms using that strategy
- No blocking, no staleness

### 3. **Biological Plausibility**
This mirrors how brains actually work:
- **Prefrontal cortex**: Strategic planning (slow, deep)
- **Motor cortex**: Reactive execution (fast, simple)
- **Basal ganglia**: Action selection within strategy

## Implementation Approach

### Phase 1: Abstract Goal Representation
```python
class AbstractGoal:
    """High-level objective that can be achieved many ways."""
    goal_type: str  # "reach", "avoid", "maintain", "explore"
    target_feature: str  # What sensor/pattern to focus on
    threshold: float  # Success criteria
    priority: float  # Importance relative to other goals
```

### Phase 2: Strategy Generation
```python
def generate_strategy(self, field_state, context):
    """Use GPU to search strategy space, not action space."""
    
    # Instead of simulating exact futures, evaluate strategies
    strategies = self.create_candidate_strategies()
    
    for strategy in strategies:
        # Simulate abstract outcomes
        value = self.evaluate_strategy_value(strategy, field_state)
        robustness = self.test_strategy_robustness(strategy)
        
    return best_strategy
```

### Phase 3: Reactive Interpreter
```python
class StrategyExecutor:
    """Converts abstract strategy to concrete actions."""
    
    def __init__(self):
        self.behavioral_primitives = {
            "approach": self.approach_behavior,
            "avoid": self.avoid_behavior,
            "explore": self.explore_behavior,
            "patrol": self.patrol_behavior
        }
    
    def execute(self, state, strategy):
        """Fast reactive execution of slow strategic plan."""
        active_goal = strategy.get_priority_goal(state)
        behavior = self.behavioral_primitives[active_goal.type]
        return behavior(state, active_goal)
```

## Benefits Over Current Approach

| Aspect | Current (Low-Level) | Pivoted (Strategic) |
|--------|-------------------|-------------------|
| Plan validity | ~100ms | 10-60 seconds |
| Cache effectiveness | 0% | 80%+ |
| Biological realism | Low | High |
| Robustness | Fragile | Adaptive |
| Decoupling | Fake | Real |

## The Beautiful Part

The GPU "deep think" now creates **wisdom**, not just predictions:
- It discovers strategies through simulation
- It identifies valuable behavioral patterns
- It learns abstract policies

The reactive brain provides **agility**:
- It executes strategies adaptively
- It handles real-time variations
- It maintains responsiveness

Together, they create intelligent behavior that's both thoughtful and responsive.

## Next Steps

1. Redefine `SimulatedAction` → `StrategicGoal`
2. Transform `GPUFutureSimulator` → `GPUStrategicPlanner`
3. Create `StrategyExecutor` for reactive interpretation
4. Test with simple strategies first

This isn't just fixing the cache problem - it's building a fundamentally better cognitive architecture.