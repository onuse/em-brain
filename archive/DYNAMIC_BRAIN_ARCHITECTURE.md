# Dynamic Brain Architecture

## Overview

The dynamic brain architecture creates brain instances on-demand based on robot capabilities, replacing the previous static configuration approach. Brains are created only when robots connect, with dimensions calculated based on robot complexity.

## Implementation Status

✅ **Completed**: The dynamic brain architecture has been successfully implemented with all core principles achieved:
- Lazy brain initialization when clients connect
- True dynamic dimensions based on robot capabilities
- Multi-client support with brain pooling
- Clean separation of concerns through interfaces

## Key Components

### 1. Clean Interface Layer (`src/core/interfaces.py`)

**IBrain** - Pure brain interface that knows nothing about robots:
- `process_field_dynamics(field_input) -> field_output`
- `get_field_dimensions() -> int`
- `get_state() -> Dict`
- `load_state(state)`

**IRobotRegistry** - Manages robot profiles:
- `register_robot(capabilities) -> Robot`
- `get_robot_profile(profile_key) -> Robot`

**IBrainPool** - Manages brain instances:
- `get_brain_for_profile(profile_key) -> IBrain`
- `get_brain_config(profile_key) -> Dict`

**IBrainService** - Manages sessions:
- `create_session(robot) -> IBrainSession`
- `close_session(session_id)`

**IAdapterFactory** - Creates robot-brain translators:
- `create_sensory_adapter(robot, field_dims) -> ISensoryAdapter`
- `create_motor_adapter(robot, field_dims) -> IMotorAdapter`

### 2. Robot Registry (`src/core/robot_registry.py`)

Parses robot capabilities from handshake:
```python
capabilities = [robot_version, sensory_size, action_size, hardware_type, capabilities_mask]
```

Creates Robot objects with:
- Sensory channels (range, unit, description)
- Motor channels (range, unit, description)
- Hardware capabilities

### 3. Brain Pool (`src/core/brain_pool.py`)

Intelligent dimension calculation:
```python
def _calculate_field_dimensions(sensory_dim, motor_dim):
    base_dims = 12
    sensory_factor = ceil(log2(sensory_dim + 1)) * 3
    motor_factor = ceil(log2(motor_dim + 1)) * 2
    complexity_bonus = ceil(sqrt(sensory_dim + motor_dim))
    return base_dims + sensory_factor + motor_factor + complexity_bonus
```

Examples:
- PiCar-X (16s/5m) → 36D field
- Minimal (8s/2m) → 28D field
- Advanced (32s/8m) → 44D field

### 4. Adapters (`src/core/adapters.py`)

**SensoryAdapter**: Robot sensors → Field coordinates
- Normalizes sensor values to [-1, 1]
- Projects to field space using learned mapping
- Applies tanh activation for bounded values

**MotorAdapter**: Field state → Robot motors
- Extracts motor values from field
- Denormalizes to motor ranges
- Ensures safe motor commands

### 5. Brain Service (`src/core/brain_service.py`)

Session management:
1. Creates brain session per robot connection
2. Coordinates brain + adapters
3. Tracks statistics (cycles, timing)
4. Handles errors gracefully

### 6. Connection Handler (`src/core/connection_handler.py`)

Protocol implementation:
- Handles extended handshake with capabilities
- Routes messages to appropriate sessions
- Manages connection lifecycle

## Brain Implementations

### SimpleFieldBrain (`src/core/simple_field_brain.py`)

Clean implementation supporting dynamic dimensions:
- 4D tensor: (field_dimensions, x, y, z)
- Simple field dynamics: decay, diffusion, activation
- Direct support for IBrain interface

### UnifiedFieldBrain Wrapper (`src/core/dynamic_brain_factory.py`)

Adapts existing brain to new interface:
- DynamicBrainWrapper implements IBrain
- Translates between field and sensory/motor spaces
- Preserves complex multi-dimensional structure

## Architecture Flow

```
Robot → Handshake → Registry → Robot Profile
                                    ↓
                              Brain Pool
                                    ↓
                         Calculate Dimensions
                                    ↓
                            Create Brain
                                    ↓
Connection Handler ← Session ← Brain Service
        ↓
   Sensory Input → Adapter → Field Input → Brain
                                              ↓
                                        Field Output
                                              ↓
   Motor Output ← Adapter ← Field Output ←────┘
```

## Benefits

1. **Dynamic Sizing**: Brains scale with robot complexity
2. **Clean Separation**: Brain knows nothing about robots
3. **Lazy Creation**: Resources allocated only when needed
4. **Type Safety**: Clear interfaces throughout
5. **Extensibility**: Easy to add new brain types
6. **Testability**: Each component independently testable

## Configuration

Settings removed - dimensions determined automatically:
- No hardcoded sensory/motor dimensions
- No predetermined field dimensions
- Spatial resolution configurable but defaults to 4

## Testing

Test with multiple robot profiles:
```python
python3 server/test_simple_brain.py
```

Shows different robots getting appropriately sized brains and field evolution over cycles.

## Technical Details

### Memory Requirements

- 28D field @ 4³ resolution: ~7KB per field state
- 36D field @ 4³ resolution: ~9KB per field state  
- 44D field @ 4³ resolution: ~11KB per field state

Multiple brain instances are feasible even on limited hardware.

### Field Dimension Algorithm

The logarithmic scaling ensures:
- Small robots get simpler brains (fewer computations)
- Complex robots get richer representations
- Computational cost scales reasonably with complexity

### Adapter Architecture

The adapter pattern provides:
- Complete isolation between robot and brain spaces
- Learnable mappings that can optimize over time
- Easy addition of new sensor/motor types
- Normalization and safety guarantees