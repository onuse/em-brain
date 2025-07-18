# Brain Error Codes Documentation

## Overview

The brain communication system uses comprehensive error codes to provide clear, actionable feedback when issues occur. Each error code includes:

- **Unique numeric code** (e.g., 5.1)
- **Descriptive name** (e.g., SIMILARITY_ENGINE_FAILURE)
- **Detailed description** of what went wrong
- **Resolution guidance** for fixing the issue
- **Severity level** (INFO, WARNING, ERROR, CRITICAL)
- **Context information** for debugging

## Error Code Categories

### 1.x - Protocol Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 1.0 | UNKNOWN_MESSAGE_TYPE | Client sent unknown message type | Check client protocol version and message format |
| 1.1 | PROTOCOL_VERSION_MISMATCH | Client-server protocol version mismatch | Update client or server to matching protocol version |
| 1.2 | INVALID_MESSAGE_FORMAT | Message format does not match protocol | Check message encoding and structure |
| 1.3 | MESSAGE_TOO_LARGE | Message exceeds maximum allowed size | Reduce message size or increase server limits |

### 2.x - Communication Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 2.0 | PROTOCOL_ERROR | General protocol communication error | Check network connection and protocol implementation |
| 2.1 | CONNECTION_TIMEOUT | Client connection timed out | Check network latency and timeout settings |
| 2.2 | CLIENT_DISCONNECTED | Client disconnected unexpectedly | Check client stability and network connection |
| 2.3 | SEND_FAILED | Failed to send response to client | Check network connection and client status |

### 3.x - Input Validation Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 3.0 | EMPTY_SENSORY_INPUT | Sensory input vector is empty | Ensure sensory input contains valid data |
| 3.1 | SENSORY_INPUT_TOO_LARGE | Sensory input exceeds maximum dimensions | Reduce sensory input size or increase server limits |
| 3.2 | INVALID_SENSORY_DIMENSIONS | Sensory input has invalid dimensions | Check sensory input format and expected dimensions |
| 3.3 | SENSORY_VALUES_OUT_OF_RANGE | Sensory input values outside expected range | Normalize sensory values to [0,1] range |

### 4.x - Action Generation Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 4.0 | INVALID_ACTION_DIMENSIONS | Requested action dimensions are invalid | Check action dimension parameter |
| 4.1 | ACTION_GENERATION_FAILED | Brain failed to generate valid action | Check brain state and experience storage |
| 4.2 | ACTION_VALUES_OUT_OF_RANGE | Generated action values are out of range | Check action normalization and bounds |

### 5.x - Brain Processing Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 5.0 | BRAIN_PROCESSING_ERROR | General brain processing failure | Check brain logs for specific component failure |
| 5.1 | SIMILARITY_ENGINE_FAILURE | Similarity engine failed to process request | Check similarity engine cache and GPU status |
| 5.2 | PREDICTION_ENGINE_FAILURE | Prediction engine failed to generate prediction | Check prediction engine and pattern analysis |
| 5.3 | EXPERIENCE_STORAGE_FAILURE | Experience storage system failed | Check memory usage and storage integrity |
| 5.4 | ACTIVATION_DYNAMICS_FAILURE | Activation dynamics system failed | Check activation state and utility calculations |
| 5.5 | PATTERN_ANALYSIS_FAILURE | Pattern analysis system failed | Check GPU status and pattern discovery cache |
| 5.6 | MEMORY_PRESSURE_ERROR | System under memory pressure | Reduce cache sizes or increase available memory |
| 5.7 | GPU_PROCESSING_ERROR | GPU processing failed | Check GPU status and MPS availability |

### 6.x - System Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 6.0 | SYSTEM_OVERLOAD | System is overloaded with requests | Reduce request rate or scale system resources |
| 6.1 | INSUFFICIENT_MEMORY | System has insufficient memory | Increase available memory or reduce cache sizes |
| 6.2 | HARDWARE_FAILURE | Hardware component failure detected | Check hardware status and restart if necessary |
| 6.3 | CONFIGURATION_ERROR | System configuration error | Check configuration files and settings |

### 7.x - Learning Errors
| Code | Name | Description | Resolution |
|------|------|-------------|------------|
| 7.0 | LEARNING_SYSTEM_FAILURE | Learning system failed | Check learning components and experience quality |
| 7.1 | EXPERIENCE_CORRUPTION | Experience data is corrupted | Clear experience storage and restart learning |
| 7.2 | SIMILARITY_LEARNING_FAILURE | Similarity learning adaptation failed | Check similarity learning parameters and data |
| 7.3 | PATTERN_DISCOVERY_FAILURE | Pattern discovery system failed | Check pattern analysis cache and GPU status |

## Dual Logging System

The brain uses a dual logging approach optimized for different use cases:

### Terminal Output (Human-Readable)
Concise messages for immediate feedback:
```
❌ Client robot_001: Similarity Engine Failure - Similarity search timed out (memory: 512MB, search: 5.20s)
⚠️  Client robot_002: Memory Pressure Error - Cache eviction triggered (memory: 512MB)
```

### Log Files (Detailed)
Comprehensive error information in `logs/brain_errors.jsonl`:
```json
{
  "timestamp": "2025-07-16T12:04:23.177634",
  "client_id": "robot_001",
  "error_code": 5.1,
  "error_name": "SIMILARITY_ENGINE_FAILURE",
  "message": "Similarity search timed out",
  "description": "Similarity engine failed to process request",
  "resolution": "Check similarity engine cache and GPU status",
  "severity": "ERROR",
  "context": {"search_time": 5.2, "query_size": 16, "memory_usage_mb": 512},
  "exception": "GPU search timeout",
  "exception_type": "TimeoutError"
}
```

### Viewing Errors
Use the error viewing tool:
```bash
# View recent errors
python3 tools/view_errors.py

# View more errors
python3 tools/view_errors.py --limit 20

# Watch for new errors in real-time
python3 tools/view_errors.py --watch
```

## Severity Levels

- **🚨 CRITICAL**: System-threatening errors requiring immediate attention
- **❌ ERROR**: Significant issues that prevent normal operation
- **⚠️ WARNING**: Issues that may impact performance but don't stop operation
- **ℹ️ INFO**: Informational messages about normal events

## Error Classification

The server automatically classifies exceptions into specific error codes based on:
- Exception type (e.g., MemoryError → MEMORY_PRESSURE_ERROR)
- Exception message content (e.g., "similarity" → SIMILARITY_ENGINE_FAILURE)
- Context clues (e.g., "gpu" → GPU_PROCESSING_ERROR)

## Common Error Patterns

### Memory Pressure (5.6)
**Symptoms**: High cache evictions, slow response times, memory warnings
**Causes**: Insufficient RAM, too many concurrent requests, large cache sizes
**Solutions**: Increase memory, reduce cache sizes, limit concurrent requests

### Similarity Engine Failure (5.1)
**Symptoms**: Search timeouts, GPU errors, cache misses
**Causes**: GPU memory issues, overloaded similarity cache, MPS problems
**Solutions**: Restart GPU, reduce cache size, check MPS availability

### Prediction Engine Failure (5.2)
**Symptoms**: Pattern analysis failures, prediction timeouts
**Causes**: Insufficient experiences, GPU processing issues, pattern cache problems
**Solutions**: Check experience storage, verify GPU status, clear pattern cache

## Using Error Codes in Code

```python
from server.src.communication.error_codes import BrainErrorCode, create_brain_error, log_brain_error

# Create specific error
error = create_brain_error(
    BrainErrorCode.SIMILARITY_ENGINE_FAILURE,
    "GPU search timeout",
    context={'timeout': 5.0, 'query_size': 16}
)

# Log with client context
log_brain_error(error, "robot_001")
```

## Migration from Old System

The old generic error codes have been replaced:
- `1.0` → `UNKNOWN_MESSAGE_TYPE` (still 1.0)
- `2.0` → `PROTOCOL_ERROR` (still 2.0)
- `3.0` → `EMPTY_SENSORY_INPUT` (still 3.0)
- `4.0` → `SENSORY_INPUT_TOO_LARGE` (now 3.1)
- `5.0` → Now classified into specific 5.x codes based on actual failure

## Benefits

1. **Immediate Understanding**: Error names are self-explanatory
2. **Faster Debugging**: Context and resolution suggestions speed up fixes
3. **Better Monitoring**: Severity levels enable proper alerting
4. **Searchable**: Error names can be easily searched in logs and documentation
5. **Maintainable**: Adding new error types is straightforward and documented