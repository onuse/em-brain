# Biological Multi-Timescale Field Proposal

## The Problem
Current unified field uses a single decay rate (0.999) for all memories. This can't handle both 30-second tests and year-long operation - it's like trying to build working memory and permanent memory from the same molecular mechanism.

## Biological Solution
Biology uses **different physical substrates** for different timescales:

### 1. Immediate Field (Phosphorylation-like)
- **Timescale**: Seconds to minutes
- **Resolution**: Full spatial/temporal detail
- **Decay**: Fast (0.95 per cycle)
- **Energy**: Minimal
- **Purpose**: Current sensory processing, immediate reactions
- **Biology analog**: Ca²⁺ dynamics, phosphorylation cascades

### 2. Working Field (Receptor-like)
- **Timescale**: Minutes to hours  
- **Resolution**: Reduced but still detailed
- **Decay**: Moderate (0.999 per cycle)
- **Energy**: Moderate (consolidation-gated)
- **Purpose**: Current task context, short-term patterns
- **Biology analog**: AMPA receptor trafficking, early LTP

### 3. Consolidated Field (Structural-like)
- **Timescale**: Days to years
- **Resolution**: Compressed, statistical
- **Decay**: Minimal (0.99999 per cycle) + baseline
- **Energy**: High (sleep-gated consolidation)
- **Purpose**: Learned skills, stable patterns
- **Biology analog**: Synaptic structural changes, late LTP

## Implementation Sketch

```python
class BiologicalUnifiedField:
    def __init__(self):
        # Three fields at different resolutions
        self.field_immediate = torch.zeros(high_res_shape)    # e.g., [20,20,20,10,15,...]
        self.field_working = torch.zeros(medium_res_shape)    # e.g., [10,10,10,5,8,...]
        self.field_consolidated = torch.zeros(low_res_shape)  # e.g., [5,5,5,3,4,...]
        
        # Biological state signals
        self.arousal = 0.5        # High = explore, Low = consolidate
        self.surprise = 0.0       # Prediction error
        self.reward = 0.0         # Value signal
        self.fatigue = 0.0        # Metabolic constraint
        
        # Consolidation parameters
        self.consolidation_threshold = 0.3
        self.consolidation_budget = 100  # Limited per cycle
        
    def process_input(self, sensory_input):
        # 1. Immediate processing (always happens)
        self.field_immediate = self._apply_input_to_field(
            self.field_immediate, sensory_input, gain=self.arousal
        )
        
        # 2. Decay immediate field (fast)
        self.field_immediate *= 0.95
        
        # 3. Gated consolidation to working memory
        importance = self.surprise + self.reward
        if importance > self.consolidation_threshold:
            self._consolidate_immediate_to_working(importance)
            
        # 4. Decay working field (slow)
        self.field_working *= 0.999
        
        # 5. Sleep-like consolidation (periodic)
        if self.is_consolidation_phase():
            self._consolidate_working_to_permanent()
            
    def _consolidate_immediate_to_working(self, importance):
        # Transfer patterns based on importance and budget
        if self.consolidation_budget > 0:
            # Downsample immediate -> working
            pattern = F.avg_pool3d(self.field_immediate[:3], 2)
            self.field_working[:3] += pattern * importance * 0.1
            self.consolidation_budget -= pattern.sum()
            
    def _consolidate_working_to_permanent(self):
        # Replay and strengthen important patterns
        # This happens during "sleep" or low-activity periods
        strong_patterns = self.field_working > 0.5
        self.field_consolidated += downsample(strong_patterns) * 0.01
```

## Key Biological Principles

### 1. **Energy Constraints**
- Consolidation has a limited "budget" per cycle
- Only important/surprising/rewarding experiences get consolidated
- Most experiences stay in immediate field and decay

### 2. **State-Dependent Processing**
- High arousal: Enhance immediate field, reduce consolidation
- Low arousal: Reduce immediate gain, increase consolidation
- Sleep state: Replay and consolidate to permanent storage

### 3. **Hierarchical Compression**
- Immediate: Full detail (what exactly happened)
- Working: Compressed patterns (what typically happens)
- Consolidated: Statistical regularities (what always happens)

### 4. **Interference as Feature**
- New patterns partially overwrite old ones in immediate/working fields
- This creates natural generalization and abstraction
- Only repeatedly important patterns make it to consolidated storage

## Advantages Over Single Decay Rate

1. **Natural timescale handling**: Each substrate optimal for its duration
2. **Energy efficient**: Most processing stays in cheap immediate field
3. **Continual learning**: Old memories protected in consolidated field
4. **Adaptive capacity**: Can be detailed when needed, compressed when not
5. **Biological plausibility**: Mirrors actual neural mechanisms

## Parameters That Work Across Timescales

```python
# Immediate field (seconds-minutes)
immediate_decay = 0.95
immediate_threshold = 0.1

# Working field (hours-days)  
working_decay = 0.999
working_baseline = 0.02
working_threshold = 0.05

# Consolidated field (months-years)
consolidated_decay = 0.99999
consolidated_baseline = 0.1
consolidated_threshold = 0.2

# These work because:
# - 30-second test: Uses immediate field primarily
# - Hour-long session: Patterns migrate to working field
# - Year-long operation: Important patterns in consolidated field
# - Each field optimal for its timescale
```

## Conclusion

Biology doesn't use one mechanism for all timescales - it uses multiple mechanisms, each optimized for its temporal range. The unified field should do the same: multiple fields with different dynamics, not one field trying to handle everything.

This isn't more complex - it's more *correct*. It's how real intelligence handles the vast range of timescales from reflexes to lifelong learning.