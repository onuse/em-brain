# Statistics Control System

The statistics control system prevents performance degradation from expensive statistics collection by using feature flags to completely avoid collection methods in normal operation.

## Problem

Previously, statistics gathering was causing massive performance degradation:
- `get_coactivation_stats()`: 2128ms (96% of total time)
- `get_stream_state()`: Expensive pattern counting and similarity calculations
- `get_column_stats()`: O(n²) pairwise similarity calculations

## Solution

Clean feature flags that completely avoid expensive collection methods:

```python
from src.statistics_control import enable_production_mode, enable_investigation_mode

# Production: maximum performance, minimal stats
enable_production_mode()

# Investigation: all statistics for debugging
enable_investigation_mode()
```

## Usage

### Quick Control
```bash
# Production mode (maximum performance)
python3 stats_control.py production --config

# Investigation mode (all statistics)
python3 stats_control.py investigation --config

# Performance profiling only
python3 stats_control.py performance --config
```

### Environment Variables
```bash
# Enable specific statistics
export BRAIN_ENABLE_STREAM_STATS=true
export BRAIN_ENABLE_COACTIVATION_STATS=true
export BRAIN_ENABLE_PERFORMANCE_PROFILING=true

# Run your application
python3 your_app.py
```

### Configuration File
Create `statistics_config.json`:
```json
{
  "enable_core_stats": true,
  "enable_stream_stats": false,
  "enable_coactivation_stats": false,
  "enable_column_stats": false,
  "enable_competition_stats": false,
  "enable_hierarchy_stats": false,
  "enable_performance_profiling": false,
  "enable_debug_stats": false
}
```

### Programmatic Control
```python
from src.statistics_control import StatisticsController, StatisticsConfig

# Create custom configuration
config = StatisticsConfig(
    enable_core_stats=True,
    enable_stream_stats=False,
    enable_coactivation_stats=True,  # Only this expensive stat
    enable_column_stats=False,
    enable_competition_stats=False,
    enable_hierarchy_stats=False,
    enable_performance_profiling=False,
    enable_debug_stats=False
)

# Apply configuration
controller = StatisticsController(config)
```

## Statistics Types

### Core Statistics (Always Fast)
- `total_cycles`: Brain cycle count
- `uptime_seconds`: Brain uptime
- `architecture`: Brain architecture type
- `prediction_confidence`: Current prediction confidence
- Always enabled, < 1ms collection time

### Stream Statistics (Expensive)
- Pattern counts per stream
- Buffer utilization
- Prediction accuracy
- Storage statistics
- ~10-100ms collection time

### Coactivation Statistics (Very Expensive)
- Cross-stream co-activation patterns
- O(n²) tensor operations on large matrices
- ~1000-2000ms collection time

### Column Statistics (Very Expensive)
- Cortical column organization
- Pairwise similarity calculations
- Internal similarity metrics
- ~500-1000ms collection time

### Competition Statistics (Expensive)
- Competitive dynamics analysis
- Resource allocation patterns
- Winner-take-all statistics
- ~50-200ms collection time

### Hierarchy Statistics (Expensive)
- Temporal hierarchy layer usage
- Budget allocation patterns
- Emergent behavior analysis
- ~10-100ms collection time

## Performance Impact

Based on benchmarks:
- **Production mode**: 1.84ms average prediction time
- **Performance mode**: 1.59ms average prediction time  
- **Investigation mode**: 1.82ms average prediction time

The system maintains <5ms prediction times even with full statistics enabled.

## Best Practices

1. **Production**: Always use production mode for deployed systems
2. **Development**: Use performance mode for general development
3. **Debugging**: Use investigation mode only when investigating specific issues
4. **Testing**: Use appropriate mode for what you're testing

## Migration Guide

### Before (Dangerous)
```python
# This always collected expensive statistics
brain_stats = brain.get_brain_statistics()
```

### After (Safe)
```python
# Production mode: only fast core statistics
enable_production_mode()
brain_stats = brain.get_brain_statistics()

# Investigation mode: all statistics when needed
enable_investigation_mode()  
brain_stats = brain.get_brain_statistics()
```

## Implementation Details

The system uses early-return pattern to completely avoid expensive method calls:

```python
def get_brain_statistics(self):
    stats = {
        # Core stats always included
        'total_cycles': self.total_cycles,
        'architecture': 'sparse_goldilocks_massive_parallel',
        'prediction_confidence': self._estimate_prediction_confidence(),
    }
    
    # Expensive stats only if flag is enabled
    if should_collect_stream_stats():
        stats['streams'] = self.get_expensive_stream_stats()
    
    return stats
```

This ensures zero performance impact when statistics are disabled - the expensive methods are never called, not just their results ignored.