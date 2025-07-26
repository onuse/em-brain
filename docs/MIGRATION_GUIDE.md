# Migration Guide: Static to Dynamic Brain Architecture

## Overview

This guide helps migrate from the old static brain architecture to the new dynamic brain architecture.

## Key Changes

### 1. Configuration Changes

**Old approach:**
```json
{
  "expected_sensory_dim": 24,
  "expected_motor_dim": 4,
  "spatial_resolution": 4
}
```

**New approach:**
```json
{
  "spatial_resolution": 4
}
```

Sensory and motor dimensions are now determined from robot capabilities during handshake.

### 2. Server Initialization

**Old:**
```python
# Brain created at startup
brain_factory = BrainFactory(config)
tcp_server = TCPServer(brain_factory.brain, host, port)
```

**New:**
```python
# Components created, brain deferred
robot_registry = RobotRegistry()
brain_pool = BrainPool(DynamicBrainFactory(config))
brain_service = BrainService(brain_pool, AdapterFactory())
connection_handler = ConnectionHandler(robot_registry, brain_service)
tcp_server = TCPServer(connection_handler, host, port)
```

### 3. Client Handshake

**Old protocol:**
```python
handshake = [robot_version]  # Simple version only
```

**New protocol:**
```python
handshake = [
    robot_version,
    sensory_size,
    action_size,
    hardware_type,
    capabilities_mask
]
```

### 4. Robot Profiles

**New concept** - Robot profiles define capabilities:
```json
{
  "robot_type": "picarx",
  "sensory_mapping": {
    "dimensions": 16,
    "channels": [
      {"name": "ultrasonic", "range_min": 0, "range_max": 300, "unit": "cm"}
    ]
  },
  "action_mapping": {
    "dimensions": 5,
    "channels": [
      {"name": "left_motor", "range_min": -100, "range_max": 100, "unit": "%"}
    ]
  }
}
```

## Migration Steps

### Step 1: Update Server Code

1. Replace direct brain imports with new architecture:
```python
# Old
from brains.field.core_brain import UnifiedFieldBrain

# New
from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.connection_handler import ConnectionHandler
```

2. Update server initialization (see examples above)

### Step 2: Update Client Code

1. Extend handshake message:
```python
# Old
handshake = struct.pack('!f', 1.0)

# New
handshake = struct.pack('!fffff',
    1.0,  # version
    16.0,  # sensory dimensions
    5.0,   # motor dimensions
    1.0,   # hardware type (1 = PiCar-X)
    3.0    # capabilities (bit mask)
)
```

2. Handle extended handshake response:
```python
# Response now includes accepted dimensions
brain_version, sensory_dim, motor_dim, gpu_avail, capabilities = struct.unpack('!fffff', response)
```

### Step 3: Create Robot Profiles

For each robot type, create a profile JSON:
```bash
client_<robot>/profile.json
```

### Step 4: Remove Old Configuration

1. Remove from settings.json:
   - expected_sensory_dim
   - expected_motor_dim
   - Any robot-specific settings

2. Keep only brain-agnostic settings:
   - spatial_resolution
   - temporal_window
   - field_evolution_rate

### Step 5: Test Migration

1. Start server with new architecture
2. Connect with updated client
3. Verify brain creation logs show correct dimensions
4. Test sensory/motor data flow

## Backwards Compatibility

To support old clients during transition:

1. Check handshake length:
```python
if len(capabilities) == 1:
    # Old client - use defaults
    sensory_dim = 24
    motor_dim = 4
else:
    # New client - use provided dimensions
    sensory_dim = int(capabilities[1])
    motor_dim = int(capabilities[2])
```

2. Use feature flag:
```python
USE_DYNAMIC_BRAIN = config.get('use_dynamic_brain', True)
```

## Common Issues

### Issue: "Brain already exists" errors
**Solution**: Remove singleton brain pattern, use brain pool

### Issue: Dimension mismatch errors
**Solution**: Ensure client sends correct dimensions in handshake

### Issue: Missing robot profile
**Solution**: Create profile JSON or use defaults from registry

### Issue: Performance regression
**Solution**: Check spatial_resolution isn't too high (keep at 4)

## Benefits After Migration

1. **Any robot works** - No server changes needed for new robots
2. **Optimal performance** - Brain complexity matches robot
3. **Multi-robot support** - Different robots on same server
4. **Cleaner code** - Better separation of concerns
5. **Future proof** - Easy to add new features

## Example: Full Migration

See `server/test_simple_brain.py` for complete example of new architecture in action.