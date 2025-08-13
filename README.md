# Em-Brain: Emergent Field Intelligence

A pure field-based artificial life system where intelligence emerges from continuous tensor dynamics without reward signals or external goals. The brain discovers meaning through experience as patterns self-organize within a unified 4D field.

## Philosophy

**No Reward Signals**: The brain discovers what is good or bad through pure experience. Like a river finding its path to the sea, behaviors emerge from intrinsic field dynamics rather than external optimization targets.

**Everything is a Field**: Sensory input, memory, motor output, and even the learning rules themselves exist as patterns within a single continuous tensor field. There are no separate modules - just regions of specialized dynamics that emerge through experience.

**Test Infrastructure, Not Intelligence**: The system includes comprehensive tests for mathematical properties and safety guarantees, but deliberately avoids testing for specific behaviors. Intelligence should surprise us.

## Key Innovations

### 1. Prediction as Core Intelligence
The system implements five phases of predictive intelligence:
- **Sensory Prediction**: Anticipating future inputs (44% accuracy on patterns)
- **Error-Driven Learning**: All learning emerges from prediction errors
- **Hierarchical Timescales**: Multi-scale predictions (97% short-term accuracy)
- **Actions as Experiments**: Motor outputs test predictive hypotheses
- **Active Sensing**: Attention follows uncertainty to maximize learning

### 2. Self-Modifying Field Dynamics
The field encodes its own evolution rules:
- **Adaptive Plasticity**: Learning rates emerge from local success
- **Regional Specialization**: Different areas develop unique dynamics
- **Meta-Learning**: The system learns how to learn better
- **Open-Ended Evolution**: No ceiling on emergent complexity

### 3. Hierarchical Scaling
The PureFieldBrain scales from tiny embedded systems to massive research configurations:
- **hardware_constrained** (6³×64): ~131K parameters for Raspberry Pi
- **small** (16³×96): ~2.5M parameters for CPU development  
- **medium** (32³×128): ~20M parameters for GPU experiments
- **massive** (64³×256): ~402M parameters for future research

### 4. Autonomous Operation Through Intrinsic Drives
- **No External Rewards**: Behavior emerges from field tensions
- **Natural Motivations**: Like water seeking its level
- **Tension-Based Planning**: Strategic patterns resolve field disequilibrium
- **True Artificial Life**: Purpose emerges from structure itself

## Architecture Overview

### Core Systems

#### Field Dynamics
- **Evolved Field Dynamics**: Self-modifying evolution rules
- **Unified Field System**: Integrates all cognitive functions
- **Spontaneous Activity**: Autonomous dynamics without input
- **Energy & Confidence**: Emerge from field state

#### Predictive Systems
- **Predictive Field System**: Generates sensory predictions
- **Hierarchical Predictions**: Multiple timescale processing
- **Action Prediction**: Selects experiments to test hypotheses
- **Prediction Error Learning**: Drives all field updates

#### Emergent Structures
- **Topology Regions**: Stable patterns form memory
- **Consolidation System**: Strengthens important patterns
- **Emergent Sensory Mapping**: Patterns self-organize
- **Strategic Patterns**: Field configurations that bias behavior

#### Active Behaviors
- **Active Sensing**: Uncertainty-driven attention
- **Motor Commands**: Emerge from field gradients
- **Pattern Library**: Learns successful behavioral strategies
- **Exploration**: Novel pattern variations and discovery

#### Autonomous Operation
- **Intrinsic Drives**: No external rewards needed
- **Field Tensions**: Natural drives from field disequilibrium
  - Information tension → exploration behavior
  - Learning tension → novelty seeking
  - Confidence tension → systematic resolution
  - Prediction tension → adaptive correction
- **Tension Resolution**: Patterns selected by how well they reduce tensions
- **True Autonomy**: Purpose emerges from structure, not external goals

### Information Flow

```
Sensory Input → Field Dynamics → Pattern Recognition
       ↓              ↓                    ↓
Prediction Error ← Predictions → Strategic Patterns
       ↓              ↓                    ↓
Field Updates ← Self-Modification → Motor Gradients
       ↓              ↓                    ↓
Pattern Library ← Behavioral Learning → Motor Output
```

## Technical Specifications

### Architecture
- **Core Implementation**: PureFieldBrain with hierarchical scaling
- **GPU Optimizations**: Grouped convolutions, batched operations, pre-allocated buffers
- **Communication**: Binary TCP protocol on port 9999
- **No Reward Signals**: Brain discovers value through experience

### Performance
- **Cycle Time Target**: <1ms on GPU (hardware_constrained config)
- **Optimization Gains**: 18x speedup for diffusion, 113x for buffer operations  
- **Memory**: Scales with configuration (131KB to 402MB+)
- **Zero Allocations**: Pre-allocated buffers in hot paths

### Testing Framework
- **Unit Tests**: Mathematical properties, tensor operations
- **Integration Tests**: Component communication, protocol integrity
- **Safety Tests**: Motor bounds, NaN handling, memory stability
- **Behavioral Tests**: Emergence validation (not specific outcomes)

## Quick Start

```bash
# Install minimal dependencies
pip install numpy torch psutil

# Install full dependencies (includes demos and robot client)
pip install -r server/requirements.txt
pip install -r client_picarx/requirements.txt

# Start brain server (safe mode for robot)
python3 server/brain.py --safe-mode

# Start robot client (on Raspberry Pi)
cd client_picarx
python3 picarx_robot.py --brain-host <SERVER-IP>

# Run behavioral tests
python3 server/tools/testing/behavioral_test_fast.py

# Run unit tests
python3 tests/unit/test_mathematical_properties.py
python3 tests/unit/test_core_infrastructure.py
```

## Project Structure

```
em-brain/
├── server/
│   ├── src/brains/field/
│   │   ├── pure_field_brain.py         # Core brain implementation
│   │   └── optimized_field_ops.py      # GPU-optimized operations
│   ├── brain.py                        # Main server entry point
│   └── tools/testing/                  # Behavioral test framework
├── client_picarx/
│   ├── picarx_robot.py                 # Robot client
│   └── src/brainstem/                  # Hardware interface (under refactor)
├── tests/
│   ├── unit/                           # Infrastructure tests
│   └── brain_test_suite.py             # Comprehensive test framework
├── docs/
│   ├── ARCHITECTURE.md                 # Detailed system design
│   ├── TODO.md                         # Development roadmap
│   └── COMM_PROTOCOL.md                # Protocol specification
└── demos/                               # Simulation and visualization
```

## Design Philosophy

### Emergence Over Engineering
- Simple rules create complex behaviors
- No explicit goals or thresholds
- Continuous functions throughout
- Minimal code, maximum capability

### Biological Inspiration
- Predictive coding from neuroscience
- Energy as metabolic state
- Confidence as certainty
- Dreams as consolidation

### True Autonomy
- No fixed hyperparameters
- System determines how to learn
- Unique "personality" per brain
- Open-ended development

## Key Concepts

### Intrinsic Drives
The field operates through tensions that create natural behavior:
- **Information Tension**: Low energy → exploration drive
- **Learning Tension**: Stagnation → novelty seeking  
- **Confidence Tension**: Uncertainty → systematic resolution
- **Prediction Tension**: Errors → adaptive correction

These aren't rewards but emergent properties of the field seeking equilibrium.

### Biological Inspiration
- **Excitation-Inhibition Balance**: Natural dynamics prevent runaway activation
- **Sparse Coding**: Efficient representation through selective activation
- **Oscillatory Dynamics**: Rhythmic patterns enable temporal processing
- **Metabolic Constraints**: Energy limitations shape computation

---

See docs/ folder for technical details:
- **ARCHITECTURE.md** - Detailed system architecture
- **TODO.md** - Development roadmap