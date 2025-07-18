#!/usr/bin/env python3
"""
Brain Communication Error Codes

Well-documented error codes for brain-robot communication protocol.
Each error has a unique code, clear description, and suggested resolution.
"""

from enum import Enum
from typing import Dict, Optional
import time


class BrainErrorCode(Enum):
    """Enumeration of all possible brain error codes."""
    
    # Protocol errors (1.x)
    UNKNOWN_MESSAGE_TYPE = 1.0
    PROTOCOL_VERSION_MISMATCH = 1.1
    INVALID_MESSAGE_FORMAT = 1.2
    MESSAGE_TOO_LARGE = 1.3
    
    # Communication errors (2.x)
    PROTOCOL_ERROR = 2.0
    CONNECTION_TIMEOUT = 2.1
    CLIENT_DISCONNECTED = 2.2
    SEND_FAILED = 2.3
    
    # Input validation errors (3.x)
    EMPTY_SENSORY_INPUT = 3.0
    SENSORY_INPUT_TOO_LARGE = 3.1
    INVALID_SENSORY_DIMENSIONS = 3.2
    SENSORY_VALUES_OUT_OF_RANGE = 3.3
    
    # Action generation errors (4.x)
    INVALID_ACTION_DIMENSIONS = 4.0
    ACTION_GENERATION_FAILED = 4.1
    ACTION_VALUES_OUT_OF_RANGE = 4.2
    
    # Brain processing errors (5.x)
    BRAIN_PROCESSING_ERROR = 5.0
    SIMILARITY_ENGINE_FAILURE = 5.1
    PREDICTION_ENGINE_FAILURE = 5.2
    EXPERIENCE_STORAGE_FAILURE = 5.3
    ACTIVATION_DYNAMICS_FAILURE = 5.4
    PATTERN_ANALYSIS_FAILURE = 5.5
    MEMORY_PRESSURE_ERROR = 5.6
    GPU_PROCESSING_ERROR = 5.7
    
    # System errors (6.x)
    SYSTEM_OVERLOAD = 6.0
    INSUFFICIENT_MEMORY = 6.1
    HARDWARE_FAILURE = 6.2
    CONFIGURATION_ERROR = 6.3
    
    # Learning errors (7.x)
    LEARNING_SYSTEM_FAILURE = 7.0
    EXPERIENCE_CORRUPTION = 7.1
    SIMILARITY_LEARNING_FAILURE = 7.2
    PATTERN_DISCOVERY_FAILURE = 7.3


class BrainError:
    """Comprehensive brain error with code, message, and context."""
    
    def __init__(self, 
                 error_code: BrainErrorCode,
                 message: str,
                 context: Optional[Dict] = None,
                 exception: Optional[Exception] = None):
        self.error_code = error_code
        self.message = message
        self.context = context or {}
        self.exception = exception
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict:
        """Convert error to dictionary for logging."""
        return {
            'error_code': self.error_code.value,
            'error_name': self.error_code.name,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp,
            'exception': str(self.exception) if self.exception else None
        }
    
    def __str__(self) -> str:
        """Human-readable error string."""
        return f"BrainError {self.error_code.value} ({self.error_code.name}): {self.message}"


class BrainErrorHandler:
    """Centralized error handling and logging for brain system."""
    
    def __init__(self):
        self.error_descriptions = {
            # Protocol errors
            BrainErrorCode.UNKNOWN_MESSAGE_TYPE: {
                'description': 'Client sent unknown message type',
                'resolution': 'Check client protocol version and message format',
                'severity': 'WARNING'
            },
            BrainErrorCode.PROTOCOL_VERSION_MISMATCH: {
                'description': 'Client-server protocol version mismatch',
                'resolution': 'Update client or server to matching protocol version',
                'severity': 'ERROR'
            },
            BrainErrorCode.INVALID_MESSAGE_FORMAT: {
                'description': 'Message format does not match protocol specification',
                'resolution': 'Check message encoding and structure',
                'severity': 'WARNING'
            },
            BrainErrorCode.MESSAGE_TOO_LARGE: {
                'description': 'Message exceeds maximum allowed size',
                'resolution': 'Reduce message size or increase server limits',
                'severity': 'WARNING'
            },
            
            # Communication errors
            BrainErrorCode.PROTOCOL_ERROR: {
                'description': 'General protocol communication error',
                'resolution': 'Check network connection and protocol implementation',
                'severity': 'ERROR'
            },
            BrainErrorCode.CONNECTION_TIMEOUT: {
                'description': 'Client connection timed out',
                'resolution': 'Check network latency and timeout settings',
                'severity': 'WARNING'
            },
            BrainErrorCode.CLIENT_DISCONNECTED: {
                'description': 'Client disconnected unexpectedly',
                'resolution': 'Check client stability and network connection',
                'severity': 'INFO'
            },
            
            # Input validation errors
            BrainErrorCode.EMPTY_SENSORY_INPUT: {
                'description': 'Sensory input vector is empty',
                'resolution': 'Ensure sensory input contains valid data',
                'severity': 'WARNING'
            },
            BrainErrorCode.SENSORY_INPUT_TOO_LARGE: {
                'description': 'Sensory input exceeds maximum dimensions',
                'resolution': 'Reduce sensory input size or increase server limits',
                'severity': 'WARNING'
            },
            BrainErrorCode.INVALID_SENSORY_DIMENSIONS: {
                'description': 'Sensory input has invalid dimensions',
                'resolution': 'Check sensory input format and expected dimensions',
                'severity': 'WARNING'
            },
            BrainErrorCode.SENSORY_VALUES_OUT_OF_RANGE: {
                'description': 'Sensory input values outside expected range',
                'resolution': 'Normalize sensory values to [0,1] range',
                'severity': 'WARNING'
            },
            
            # Action generation errors
            BrainErrorCode.INVALID_ACTION_DIMENSIONS: {
                'description': 'Requested action dimensions are invalid',
                'resolution': 'Check action dimension parameter',
                'severity': 'ERROR'
            },
            BrainErrorCode.ACTION_GENERATION_FAILED: {
                'description': 'Brain failed to generate valid action',
                'resolution': 'Check brain state and experience storage',
                'severity': 'ERROR'
            },
            BrainErrorCode.ACTION_VALUES_OUT_OF_RANGE: {
                'description': 'Generated action values are out of range',
                'resolution': 'Check action normalization and bounds',
                'severity': 'WARNING'
            },
            
            # Brain processing errors
            BrainErrorCode.BRAIN_PROCESSING_ERROR: {
                'description': 'General brain processing failure',
                'resolution': 'Check brain logs for specific component failure',
                'severity': 'ERROR'
            },
            BrainErrorCode.SIMILARITY_ENGINE_FAILURE: {
                'description': 'Similarity engine failed to process request',
                'resolution': 'Check similarity engine cache and GPU status',
                'severity': 'ERROR'
            },
            BrainErrorCode.PREDICTION_ENGINE_FAILURE: {
                'description': 'Prediction engine failed to generate prediction',
                'resolution': 'Check prediction engine and pattern analysis',
                'severity': 'ERROR'
            },
            BrainErrorCode.EXPERIENCE_STORAGE_FAILURE: {
                'description': 'Experience storage system failed',
                'resolution': 'Check memory usage and storage integrity',
                'severity': 'ERROR'
            },
            BrainErrorCode.ACTIVATION_DYNAMICS_FAILURE: {
                'description': 'Activation dynamics system failed',
                'resolution': 'Check activation state and utility calculations',
                'severity': 'ERROR'
            },
            BrainErrorCode.PATTERN_ANALYSIS_FAILURE: {
                'description': 'Pattern analysis system failed',
                'resolution': 'Check GPU status and pattern discovery cache',
                'severity': 'ERROR'
            },
            BrainErrorCode.MEMORY_PRESSURE_ERROR: {
                'description': 'System under memory pressure',
                'resolution': 'Reduce cache sizes or increase available memory',
                'severity': 'WARNING'
            },
            BrainErrorCode.GPU_PROCESSING_ERROR: {
                'description': 'GPU processing failed',
                'resolution': 'Check GPU status and MPS availability',
                'severity': 'ERROR'
            },
            
            # System errors
            BrainErrorCode.SYSTEM_OVERLOAD: {
                'description': 'System is overloaded with requests',
                'resolution': 'Reduce request rate or scale system resources',
                'severity': 'WARNING'
            },
            BrainErrorCode.INSUFFICIENT_MEMORY: {
                'description': 'System has insufficient memory',
                'resolution': 'Increase available memory or reduce cache sizes',
                'severity': 'ERROR'
            },
            BrainErrorCode.HARDWARE_FAILURE: {
                'description': 'Hardware component failure detected',
                'resolution': 'Check hardware status and restart if necessary',
                'severity': 'CRITICAL'
            },
            BrainErrorCode.CONFIGURATION_ERROR: {
                'description': 'System configuration error',
                'resolution': 'Check configuration files and settings',
                'severity': 'ERROR'
            },
            
            # Learning errors
            BrainErrorCode.LEARNING_SYSTEM_FAILURE: {
                'description': 'Learning system failed',
                'resolution': 'Check learning components and experience quality',
                'severity': 'ERROR'
            },
            BrainErrorCode.EXPERIENCE_CORRUPTION: {
                'description': 'Experience data is corrupted',
                'resolution': 'Clear experience storage and restart learning',
                'severity': 'ERROR'
            },
            BrainErrorCode.SIMILARITY_LEARNING_FAILURE: {
                'description': 'Similarity learning adaptation failed',
                'resolution': 'Check similarity learning parameters and data',
                'severity': 'WARNING'
            },
            BrainErrorCode.PATTERN_DISCOVERY_FAILURE: {
                'description': 'Pattern discovery system failed',
                'resolution': 'Check pattern analysis cache and GPU status',
                'severity': 'WARNING'
            }
        }
    
    def create_error(self, 
                    error_code: BrainErrorCode,
                    message: str = None,
                    context: Dict = None,
                    exception: Exception = None) -> BrainError:
        """Create a comprehensive brain error."""
        if message is None:
            message = self.error_descriptions[error_code]['description']
        
        return BrainError(error_code, message, context, exception)
    
    def get_error_info(self, error_code: BrainErrorCode) -> Dict:
        """Get detailed error information."""
        return self.error_descriptions.get(error_code, {
            'description': 'Unknown error code',
            'resolution': 'Check error code documentation',
            'severity': 'ERROR'
        })
    
    def log_error(self, brain_error: BrainError, client_id: str = None) -> None:
        """Log error with dual output: human-readable terminal + detailed file."""
        error_info = self.get_error_info(brain_error.error_code)
        severity = error_info['severity']
        
        # 1. Terminal output - concise and human-readable
        terminal_msg = self._format_terminal_message(brain_error, error_info, client_id)
        self._log_to_terminal(terminal_msg, severity)
        
        # 2. File output - detailed information
        self._log_to_file(brain_error, error_info, client_id)
    
    def _format_terminal_message(self, brain_error: BrainError, error_info: dict, client_id: str = None) -> str:
        """Format concise message for terminal display."""
        # Create human-readable error message
        error_name = brain_error.error_code.name.replace('_', ' ').title()
        
        # Add client context if available
        client_part = f"Client {client_id}: " if client_id else ""
        
        # Core message
        core_msg = f"{client_part}{error_name} - {brain_error.message}"
        
        # Add key context if available
        if brain_error.context:
            key_context = self._extract_key_context(brain_error.context)
            if key_context:
                core_msg += f" ({key_context})"
        
        return core_msg
    
    def _extract_key_context(self, context: dict) -> str:
        """Extract most relevant context for terminal display."""
        key_items = []
        
        # Priority context items for terminal
        if 'input_size' in context:
            key_items.append(f"size: {context['input_size']}")
        if 'memory_usage_mb' in context:
            key_items.append(f"memory: {context['memory_usage_mb']}MB")
        if 'timeout' in context:
            key_items.append(f"timeout: {context['timeout']}s")
        if 'search_time' in context:
            key_items.append(f"search: {context['search_time']:.2f}s")
        if 'requests_served' in context:
            key_items.append(f"requests: {context['requests_served']}")
        
        return ", ".join(key_items) if key_items else ""
    
    def _log_to_terminal(self, message: str, severity: str) -> None:
        """Log concise message to terminal."""
        if severity == 'CRITICAL':
            print(f"🚨 {message}")
        elif severity == 'ERROR':
            print(f"❌ {message}")
        elif severity == 'WARNING':
            print(f"⚠️  {message}")
        else:
            print(f"ℹ️  {message}")
    
    def _log_to_file(self, brain_error: BrainError, error_info: dict, client_id: str = None) -> None:
        """Log detailed error information to file."""
        import json
        from datetime import datetime
        from pathlib import Path
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create error log file path
        error_log_file = logs_dir / "brain_errors.jsonl"
        
        # Prepare detailed log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'error_code': brain_error.error_code.value,
            'error_name': brain_error.error_code.name,
            'message': brain_error.message,
            'description': error_info['description'],
            'resolution': error_info['resolution'],
            'severity': error_info['severity'],
            'context': brain_error.context,
            'exception': str(brain_error.exception) if brain_error.exception else None,
            'exception_type': type(brain_error.exception).__name__ if brain_error.exception else None
        }
        
        # Append to error log file
        try:
            with open(error_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"⚠️  Failed to write error log: {e}")
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors from log file."""
        import json
        from pathlib import Path
        
        error_log_file = Path("logs/brain_errors.jsonl")
        if not error_log_file.exists():
            return []
        
        try:
            errors = []
            with open(error_log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    errors.append(json.loads(line.strip()))
            return errors
        except Exception as e:
            print(f"⚠️  Failed to read error log: {e}")
            return []
    
    def get_error_code_by_name(self, name: str) -> Optional[BrainErrorCode]:
        """Get error code by name for easy lookup."""
        try:
            return BrainErrorCode[name.upper()]
        except KeyError:
            return None


# Global error handler instance
brain_error_handler = BrainErrorHandler()


def create_brain_error(error_code: BrainErrorCode, 
                      message: str = None,
                      context: Dict = None,
                      exception: Exception = None) -> BrainError:
    """Convenience function to create brain error."""
    return brain_error_handler.create_error(error_code, message, context, exception)


def log_brain_error(brain_error: BrainError, client_id: str = None) -> None:
    """Convenience function to log brain error."""
    brain_error_handler.log_error(brain_error, client_id)