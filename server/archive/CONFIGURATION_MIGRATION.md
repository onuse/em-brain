# Configuration System Migration Guide

## Current Situation (Too Many Configs!)

We currently have **6+ different configuration sources**:

1. `settings.json` - Original config (spatial_resolution: 50)
2. `settings_simple.json` - Simplified config  
3. `settings_field_brain.json` - Field brain specific (spatial_resolution: 20)
4. `hardware_defaults.json` - Hardware scaling factors
5. `developer_overrides.json` - Testing overrides (force_spatial_resolution: 4)
6. `adaptive_config.py` - Complex 3-tier system
7. `hardware_adaptation.py` - Runtime adaptation
8. Hardcoded defaults in various files

## Proposed Solution: One Unified System

### 1. Single Configuration File: `settings_unified.json`

```json
{
  "brain": {
    "type": "field",
    "sensory_dim": 24,
    "motor_dim": 4,
    
    "// spatial_resolution": "null = auto, or specify 3-8",
    "spatial_resolution": null,
    
    "// features": "Enable/disable features",
    "enhanced_dynamics": true,
    "attention_guidance": true,
    "hierarchical_processing": true
  },
  
  "performance": {
    "target_cycle_time_ms": 150,
    
    "// overrides": "Force values (development only)",
    "force_spatial_resolution": 4,
    "force_cycle_time_ms": null
  }
}
```

### 2. Single Hardware System: `hardware_unified.py`

Combines:
- Hardware detection (CPU/GPU/Memory)
- Performance benchmarking
- Initial configuration
- Runtime adaptation
- GPU usage decisions

### 3. Configuration Hierarchy

```
1. Command-line args (highest priority)
   python brain_server.py --spatial-resolution 3

2. Environment variables
   BRAIN_SPATIAL_RESOLUTION=3 python brain_server.py

3. settings_unified.json
   - User settings
   - Performance overrides

4. Hardware auto-detection (lowest priority)
   - Benchmark-based defaults
   - Runtime adaptation
```

## Migration Steps

### Phase 1: Immediate Fix (Current)
- Keep existing system but force spatial_resolution: 4 in developer_overrides.json
- This stops the 60³ catastrophe

### Phase 2: Consolidation
1. Create `settings_unified.json` with essential settings only
2. Update `brain_server.py` to use unified config
3. Deprecate other config files

### Phase 3: Full Migration
1. Replace `adaptive_config.py` with `hardware_unified.py`
2. Remove all other config files
3. Update all references

## Benefits

1. **Single source of truth** for each setting
2. **Clear hierarchy** - no more guessing which config wins
3. **Simpler debugging** - one place to check
4. **Better performance** - no more accidental 60³ tensors
5. **Easier deployment** - one config file to manage

## Example Usage

### Default (Auto-adapt)
```json
{
  "brain": {
    "spatial_resolution": null  // Hardware will decide
  }
}
```

### Development (Force low resolution)
```json
{
  "performance": {
    "force_spatial_resolution": 3  // Override for slow hardware
  }
}
```

### Production (Explicit setting)
```json
{
  "brain": {
    "spatial_resolution": 5  // Explicit choice
  }
}
```

## Current Status

- ✅ Created `settings_unified.json`
- ✅ Created `hardware_unified.py` 
- ✅ Updated brain_server.py to support unified config
- ⏳ Need to migrate all code to use new system
- ⏳ Need to deprecate old config files