# Demo Consolidation Plan

## Problem
We have multiple 2D world demos that show different subsets of features, creating confusion about which one represents the "complete" system.

## Solution
**ONE canonical 2D demo** with ALL features integrated.

## New Structure

### Primary Demo (THE definitive showcase)
- **`demo_ultimate_2d_brain.py`** - THE complete 2D brain system
  - All drives (survival, curiosity, exploration, goal generation)
  - Adaptive parameter tuning
  - Universal actuator discovery  
  - Persistent memory across sessions
  - Visual brain state monitoring
  - Cross-session learning accumulation
  - **This is the only 2D demo most users need**

### Specialized Demos (for specific purposes)
- **`demo_distributed_brain.py`** - Multi-robot distributed architecture
- **`demo_world_agnostic_brain.py`** - Different robot embodiments
- **`demo_phase1.py`** - Core architecture reference

### Deprecated/Redundant (move to deprecated/)
- `demo_complete_brain.py` - replaced by ultimate version
- `demo_curiosity_2d_world.py` - curiosity now integrated in ultimate
- `demo_goal_driven.py` - goal generation now integrated in ultimate
- `demo_visualization.py` - visualization now integrated in ultimate

## Benefits
1. **Clear entry point**: Users know exactly which demo to run
2. **No feature fragmentation**: Everything in one place
3. **Less maintenance**: One canonical implementation
4. **Better testing**: All features tested together
5. **Clearer documentation**: One demo to document and explain

## User Experience
**First-time users:**
```bash
python3 demo_ultimate_2d_brain.py
```
Gets them the complete experience immediately.

**Advanced users:**
Can explore specialized demos for specific research interests.