"""
Error Codes and Classification for Dynamic Brain Architecture

Provides standardized error codes and handling for the brain server.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
import time


class ErrorCategory(Enum):
    """Error categories for classification."""
    CONNECTION = "CONNECTION"      # Network/connection issues
    HANDSHAKE = "HANDSHAKE"       # Handshake protocol errors
    SESSION = "SESSION"           # Session management errors
    BRAIN = "BRAIN"              # Brain processing errors
    PERSISTENCE = "PERSISTENCE"   # Storage/persistence errors
    RESOURCE = "RESOURCE"        # Resource limitation errors
    PROTOCOL = "PROTOCOL"        # Protocol violation errors
    

class ErrorCode(Enum):
    """Standardized error codes."""
    # Connection errors (1xxx)
    CONNECTION_REFUSED = 1001
    CONNECTION_TIMEOUT = 1002
    CONNECTION_LOST = 1003
    
    # Handshake errors (2xxx)
    INVALID_CAPABILITIES = 2001
    UNSUPPORTED_ROBOT = 2002
    HANDSHAKE_TIMEOUT = 2003
    DUPLICATE_SESSION = 2004
    
    # Session errors (3xxx)
    SESSION_NOT_FOUND = 3001
    SESSION_EXPIRED = 3002
    SESSION_LIMIT_REACHED = 3003
    
    # Brain errors (4xxx)
    BRAIN_CREATION_FAILED = 4001
    BRAIN_PROCESSING_ERROR = 4002
    BRAIN_OVERLOAD = 4003
    FIELD_DIMENSION_ERROR = 4004
    
    # Persistence errors (5xxx)
    PERSISTENCE_SAVE_FAILED = 5001
    PERSISTENCE_LOAD_FAILED = 5002
    PERSISTENCE_CORRUPTION = 5003
    
    # Resource errors (6xxx)
    MEMORY_LIMIT_EXCEEDED = 6001
    CPU_OVERLOAD = 6002
    GPU_UNAVAILABLE = 6003
    
    # Protocol errors (7xxx)
    INVALID_MESSAGE_FORMAT = 7001
    MESSAGE_TOO_LARGE = 7002
    PROTOCOL_VERSION_MISMATCH = 7003
    

class BrainError(Exception):
    """Base exception for brain-related errors."""
    
    def __init__(self, code: ErrorCode, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        self.category = self._determine_category(code)
        super().__init__(self._format_message())
        
    def _determine_category(self, code: ErrorCode) -> ErrorCategory:
        """Determine error category from code."""
        code_value = code.value
        if 1000 <= code_value < 2000:
            return ErrorCategory.CONNECTION
        elif 2000 <= code_value < 3000:
            return ErrorCategory.HANDSHAKE
        elif 3000 <= code_value < 4000:
            return ErrorCategory.SESSION
        elif 4000 <= code_value < 5000:
            return ErrorCategory.BRAIN
        elif 5000 <= code_value < 6000:
            return ErrorCategory.PERSISTENCE
        elif 6000 <= code_value < 7000:
            return ErrorCategory.RESOURCE
        elif 7000 <= code_value < 8000:
            return ErrorCategory.PROTOCOL
        else:
            return ErrorCategory.BRAIN
            
    def _format_message(self) -> str:
        """Format error message with code."""
        return f"[{self.code.name}] {self.message}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error': True,
            'code': self.code.value,
            'code_name': self.code.name,
            'category': self.category.value,
            'message': self.message,
            'details': self.details
        }
        
    def to_response(self) -> List[float]:
        """Convert to error response for protocol."""
        # Error response format: [-1.0, error_code, category, 0.0, 0.0]
        return [
            -1.0,  # Error indicator
            float(self.code.value),
            float(list(ErrorCategory).index(self.category)),
            0.0,
            0.0
        ]


class ErrorHandler:
    """Handles and tracks errors across the system."""
    
    def __init__(self):
        self.error_counts: Dict[ErrorCode, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 100
        
    def handle_error(self, error: BrainError, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle and track an error."""
        # Update counts
        self.error_counts[error.code] = self.error_counts.get(error.code, 0) + 1
        
        # Record error
        error_record = {
            'timestamp': time.time(),
            'client_id': client_id,
            **error.to_dict()
        }
        
        self.recent_errors.append(error_record)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
            
        return error_record
        
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        # Group by category
        category_counts = {}
        for code, count in self.error_counts.items():
            category = self._get_category(code)
            category_counts[category.value] = category_counts.get(category.value, 0) + count
            
        return {
            'total_errors': sum(self.error_counts.values()),
            'by_code': {code.name: count for code, count in self.error_counts.items()},
            'by_category': category_counts,
            'recent_errors': self.recent_errors[-10:]  # Last 10 errors
        }
        
    def _get_category(self, code: ErrorCode) -> ErrorCategory:
        """Get category for error code."""
        dummy_error = BrainError(code, "")
        return dummy_error.category


# Convenience functions for common errors
def handshake_error(message: str, capabilities: List[float] = None) -> BrainError:
    """Create handshake error."""
    return BrainError(
        ErrorCode.INVALID_CAPABILITIES,
        message,
        {'capabilities': capabilities} if capabilities else {}
    )

def session_not_found(client_id: str) -> BrainError:
    """Create session not found error."""
    return BrainError(
        ErrorCode.SESSION_NOT_FOUND,
        f"No active session for client {client_id}",
        {'client_id': client_id}
    )

def brain_processing_error(message: str, details: Dict[str, Any] = None) -> BrainError:
    """Create brain processing error."""
    return BrainError(
        ErrorCode.BRAIN_PROCESSING_ERROR,
        message,
        details
    )

def resource_error(message: str, resource_type: str, current_usage: float) -> BrainError:
    """Create resource error."""
    code = ErrorCode.MEMORY_LIMIT_EXCEEDED if resource_type == "memory" else ErrorCode.CPU_OVERLOAD
    return BrainError(
        code,
        message,
        {'resource_type': resource_type, 'current_usage': current_usage}
    )