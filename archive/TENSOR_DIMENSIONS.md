# Tensor Dimension Specification & Standards

## The Problem

Tensor dimension mismatches are causing repeated system failures across different contexts:

- **Server vs Client**: Different assumptions about action vector sizes (8D vs 4D)
- **Brain Architectures**: Different temporal dimensions (2D vs 4D) causing runtime errors
- **Protocol Evolution**: No clear standard for what dimensions mean or how to extend them
- **Context-Dependent**: Same brain works in one context but fails in another

**Root Cause**: No authoritative specification for what each dimension represents and how they should scale.

## Core Tensor Types

### 1. Sensory Vector (Input)
**Current Size**: 16D
**Purpose**: Robot sensor readings and environmental state

**Standard Breakdown**:
```
Indices 0-5:   Distance sensors (6 channels, 0-1 normalized)
Indices 6-7:   Target direction (sin/cos, -1 to +1)
Indices 8-9:   Robot velocity (x/y, -1 to +1) 
Indices 10-11: Robot orientation (sin/cos, -1 to +1)
Indices 12-13: Battery/energy state (level, temperature, 0-1)
Indices 14-15: System state (memory pressure, CPU load, 0-1)
```

**Future Extensions** (32D max):
- Visual features (8D): Edge detection, motion vectors, color histograms
- Audio features (4D): Sound direction, volume, frequency analysis
- Proprioception (4D): Joint angles, motor temperatures, torque feedback

### 2. Action Vector (Output)
**Current Size**: 4D  
**Purpose**: Robot motor commands and control signals

**Standard Breakdown**:
```
Index 0: Forward/backward velocity (-1 to +1)
Index 1: Left/right turn rate (-1 to +1)
Index 2: Auxiliary motor 1 (servo, gripper, etc., -1 to +1)
Index 3: Auxiliary motor 2 (head tilt, camera, etc., -1 to +1)
```

**Future Extensions** (8D max):
- Advanced manipulation (grip force, wrist rotation)
- Multi-robot coordination (communication flags)
- Energy management (sleep mode, performance scaling)
- Emotional expression (vocal tone, LED patterns)

### 3. Temporal Vector (Internal)
**Current Size**: 4D
**Purpose**: Biological timing and rhythm encoding

**Standard Breakdown**:
```
Index 0: Breathing rhythm (1 Hz, biological baseline)
Index 1: Alpha waves (10 Hz, attention/focus cycles)
Index 2: Circadian component (relative time 0-1 per hour)
Index 3: Long-term time (hours since start, normalized)
```

**Biological Justification**: These frequencies are fundamental to mammalian neural processing and provide temporal scaffolding for learning.

**Future Extensions** (8D max):
- Ultradian rhythms (90-minute learning cycles)
- Seasonal patterns (for long-term adaptation)
- Social synchronization (coordination with other agents)

## Dimension Compatibility Matrix

| Brain Type | Sensory | Action | Temporal | Notes |
|------------|---------|--------|----------|-------|
| MinimalBrain | 16D | 4D | 4D | Current standard |
| SparseGoldilocks | 16D | 4D | 4D | Fixed from 2D bug |
| Legacy | 8D | 2D | 2D | Archive only |
| Future Enhanced | 32D | 8D | 8D | Roadmap target |

## Protocol Standards

### 1. Handshake Specification
**Current**: `[version, sensory_size, action_size, hardware_type]`
**Problem**: No temporal dimension negotiation

**Improved**: `[version, sensory_size, action_size, temporal_size, hardware_type, capabilities_mask]`

```python
capabilities_mask bits:
0: Basic sensory-motor (all robots)
1: Visual processing capability  
2: Audio processing capability
3: Advanced manipulation capability
4: Multi-agent coordination capability
5-7: Reserved for future features
```

### 2. Auto-Negotiation Protocol
Instead of hardcoded dimensions, implement capability negotiation:

```python
class TensorNegotiation:
    def negotiate_dimensions(self, client_caps, server_caps):
        # Find maximum common dimensions
        sensory_dim = min(client_caps.max_sensory, server_caps.max_sensory)
        action_dim = min(client_caps.max_action, server_caps.max_action)
        temporal_dim = server_caps.temporal_dim  # Server determines
        
        # Ensure minimum viable dimensions
        sensory_dim = max(sensory_dim, 8)  # Minimum for basic operation
        action_dim = max(action_dim, 2)    # Forward/turn minimum
        temporal_dim = max(temporal_dim, 2) # Basic temporal processing
        
        return TensorConfig(sensory_dim, action_dim, temporal_dim)
```

### 3. Graceful Degradation
When dimension mismatches occur:
- **Truncate**: Use first N dimensions if client sends more than expected
- **Pad**: Zero-pad if client sends fewer than expected  
- **Fallback**: Drop to minimal compatible dimensions
- **Error**: Only fail if below minimum viable dimensions

## Implementation Guidelines

### 1. Brain Architecture
All brain implementations must support configurable dimensions:

```python
class Brain:
    def __init__(self, sensory_dim: int, action_dim: int, temporal_dim: int):
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim  
        self.temporal_dim = temporal_dim
        
        # No hardcoded tensor sizes anywhere
        self.sensory_stream = VectorStream(sensory_dim)
        self.action_stream = VectorStream(action_dim)
        self.temporal_stream = VectorStream(temporal_dim)
```

### 2. Client Implementation
Clients must declare their true capabilities:

```python
class RobotClient:
    def get_capabilities(self):
        return {
            'max_sensory_dim': self.sensor_count,
            'max_action_dim': self.actuator_count,
            'hardware_type': self.robot_type,
            'capabilities': self.feature_mask
        }
```

### 3. Configuration Validation
All settings must be validated at startup:

```python
def validate_tensor_config(config):
    assert config.sensory_dim >= 8, "Minimum 8D sensory for basic operation"
    assert config.action_dim >= 2, "Minimum 2D action for basic movement"  
    assert config.temporal_dim >= 2, "Minimum 2D temporal for basic timing"
    assert config.sensory_dim <= 64, "Maximum 64D sensory (hardware limit)"
    assert config.action_dim <= 16, "Maximum 16D action (hardware limit)"
    assert config.temporal_dim <= 16, "Maximum 16D temporal (biological limit)"
```

## Migration Strategy

### Phase 1: Fix Current Issues (Immediate)
- ✅ Align server motor_dim with client announcements (4D)
- ✅ Fix SparseGoldilocks temporal dimension bug
- ✅ Add dimension validation at startup

### Phase 2: Protocol Enhancement (Next Sprint)
- Implement capability negotiation handshake
- Add graceful dimension handling
- Create comprehensive test suite for all dimension combinations

### Phase 3: Advanced Features (Future)
- Auto-discovery of robot capabilities
- Dynamic dimension adjustment during runtime
- Multi-modal tensor fusion (vision + audio + proprioception)

## Testing Requirements

Every brain architecture must pass these dimension tests:

1. **Minimal Config**: 8D sensory, 2D action, 2D temporal
2. **Standard Config**: 16D sensory, 4D action, 4D temporal  
3. **Extended Config**: 32D sensory, 8D action, 8D temporal
4. **Mismatch Handling**: Graceful degradation when dimensions don't match
5. **Cross-Architecture**: Same tensor config works across all brain types

## Documentation Standards

Every tensor operation must document:
- **Input dimensions**: Expected tensor shapes
- **Output dimensions**: Resulting tensor shapes
- **Failure modes**: What happens with wrong dimensions
- **Compatibility**: Which brain architectures support this operation

## Future-Proofing Principles

1. **No Hardcoded Dimensions**: All tensor sizes must be configurable
2. **Capability Declaration**: Clients/servers declare what they support
3. **Graceful Degradation**: System continues working with reduced capability
4. **Semantic Meaning**: Each dimension index has documented meaning
5. **Biological Constraints**: Respect neurological and physical limits

This specification prevents the "competing concepts during implementation" problem by establishing a single source of truth for what each tensor dimension means and how they should be handled across the entire system.