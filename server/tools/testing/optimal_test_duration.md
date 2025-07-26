# Optimal Test Duration Analysis

Based on the behavioral test data, here's what happens at different time scales:

## Learning Phases

### 0-10 seconds: Initial Imprinting
- Brain discovers first topology regions
- Prediction confidence jumps from 0.5 to 0.999
- Field activation grows from 0.01 to ~10
- **Score improvement**: 0% → 25% (rapid)

### 10-30 seconds: Pattern Establishment  
- Multiple regions discovered (region_0, region_1, etc.)
- Field activation continues growing (10 → 100+)
- Constraint discovery begins
- **Score improvement**: 25% → 35% (moderate)

### 30-120 seconds: Refinement
- Existing patterns strengthen
- More nuanced differentiation between patterns
- Field energy stabilizes gradually
- **Score improvement**: 35% → 50% (slower)

### 120+ seconds: Convergence
- Learning rate plateaus
- Focus shifts to fine-tuning
- Minimal score improvements
- **Score improvement**: <5% per minute

## Recommended Test Durations

### Quick Validation (15-30 seconds)
- **Purpose**: Verify brain is learning
- **Expected scores**: 
  - Prediction: 0.20-0.30
  - Pattern: 0.03-0.10
  - Stability: 0.00-0.05
- **Good for**: CI/CD, quick checks

### Standard Testing (60-120 seconds)
- **Purpose**: Meaningful behavioral assessment
- **Expected scores**:
  - Prediction: 0.35-0.50
  - Pattern: 0.15-0.30
  - Stability: 0.10-0.20
- **Good for**: Development validation

### Comprehensive Testing (5-10 minutes)
- **Purpose**: Full capability assessment
- **Expected scores**:
  - Prediction: 0.60-0.80
  - Pattern: 0.40-0.60
  - Stability: 0.30-0.50
- **Good for**: Release validation

## Key Insights

1. **Prediction learning** plateaus quickly (30-60s)
2. **Pattern recognition** needs more time (120s+)
3. **Field stability** is slowest (300s+)
4. **Diminishing returns** after 2-3 minutes

## Recommendation

**120 seconds is optimal** for meaningful results because:
- Long enough to see real learning progress
- Short enough to be practical
- Captures all three learning phases
- Provides stable, reproducible scores