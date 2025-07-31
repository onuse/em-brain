# Field-Native Intelligence System

A continuous field-based artificial brain that combines predictive processing with self-modifying dynamics to create emergent intelligence. This system implements field dynamics as the substrate for cognition, where prediction serves as a fundamental organizing principle alongside other emergent properties.

## Core Approach

This project explores continuous field dynamics as an alternative to discrete AI approaches. Intelligence emerges from the topology and evolution of a 4D tensor field, where multiple cognitive principles work together:

- **Predictive Processing**: The brain continuously predicts future states and learns from errors
- **Self-Modifying Dynamics**: Evolution rules are encoded within the field itself
- **Emergent Organization**: Structure and function arise from field interactions
- **Continuous Adaptation**: No fixed parameters - everything evolves through experience

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

### 3. Unified Field Architecture
- **4D Tensor**: [32, 32, 32, 64] continuous field
- **No Semantic Encoding**: Properties emerge from dynamics
- **GPU Optimized**: Efficient tensor operations
- **Pattern-Based**: All processing through pattern matching

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
- **Reward Topology**: Goals shape field landscape

#### Active Behaviors
- **Active Sensing**: Uncertainty-driven attention
- **Motor Cortex**: Pattern-to-action translation
- **Exploration**: Emerges from low confidence
- **Dream States**: Novel pattern combinations

### Information Flow

```
Sensory Input → Field Dynamics → Pattern Recognition
       ↓              ↓                    ↓
Prediction Error ← Predictions → Action Selection
       ↓              ↓                    ↓
Field Updates ← Self-Modification → Motor Output
```

## Implementation Status

### ✅ Fully Implemented
- Complete predictive processing pipeline (5 phases)
- Self-modifying field dynamics with regional specialization
- Pattern-based cognition without coordinates
- Active vision with emergent eye movements
- Modality-agnostic sensing framework
- Cross-session persistence
- GPU acceleration (CUDA/MPS/CPU)

### 🚧 In Development
- Hardware integration (cameras, microphones)
- Multi-brain communication
- Persistence compression
- Additional sensory modalities

### 📊 Performance
- **Cycle Time**: ~250ms on M1 MacBook
- **Memory**: ~8MB field tensor
- **Scaling**: Fixed computation per cycle
- **Efficiency**: ~4% GPU utilization

## Quick Start

```bash
# Install dependencies
pip install numpy torch
pip install pygame  # Optional: for visual demo

# Run interactive demo (visual robot simulation)
python3 demo.py

# Other demo modes
python3 demo.py --mode terminal  # Terminal-only demo
python3 demo.py --mode server    # Start brain server

# Start brain server directly
python3 server/brain.py
```

## Project Structure

```
brain/
├── server/src/brains/field/    # Core implementation
│   ├── simplified_unified_brain.py     # Main brain
│   ├── evolved_field_dynamics.py       # Self-modification
│   ├── predictive_field_system.py      # Predictions
│   ├── action_prediction_system.py     # Action selection
│   ├── active_sensing_system.py        # Attention
│   └── topology_region_system.py       # Memory
├── tests/                      # Test suite
├── docs/                       # Documentation
└── tools/                      # Development utilities
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

## Research Foundation

Built on established principles:
- **Free Energy Principle**: Minimize surprise through prediction
- **Embodied Cognition**: Intelligence through interaction
- **Complex Systems**: Emergence from simple rules
- **Self-Organization**: Order from dynamics

## Future Directions

- Multi-brain networks with emergent communication
- Abstract reasoning through hierarchical predictions
- Long-term episodic memory
- Language emergence from prediction alignment

---

See docs/ folder for technical details:
- **ARCHITECTURE.md** - Detailed system architecture
- **TODO.md** - Development roadmap