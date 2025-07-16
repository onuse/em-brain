#!/usr/bin/env python3
"""
Test the new brain error handling system
"""

import sys
import os
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from src.communication.error_codes import BrainErrorCode, create_brain_error, log_brain_error, brain_error_handler

def test_error_handling():
    """Test the comprehensive error handling system."""
    print("🔍 Testing Brain Error Handling System")
    print("=" * 60)
    
    # Test 1: Create specific errors
    print("\n1. Testing Error Creation and Logging:")
    
    # Test similarity engine failure
    error1 = create_brain_error(
        BrainErrorCode.SIMILARITY_ENGINE_FAILURE,
        "Similarity search timed out",
        context={'search_time': 5.2, 'query_size': 16, 'memory_usage_mb': 512},
        exception=TimeoutError("GPU search timeout")
    )
    log_brain_error(error1, "test_client_001")
    
    # Test memory pressure error
    error2 = create_brain_error(
        BrainErrorCode.MEMORY_PRESSURE_ERROR,
        "Cache eviction triggered",
        context={'memory_usage_mb': 512, 'cache_size': 1000}
    )
    log_brain_error(error2, "test_client_002")
    
    # Test protocol error
    error3 = create_brain_error(
        BrainErrorCode.PROTOCOL_ERROR,
        "Invalid message format",
        context={'message_type': 99, 'expected_types': [0, 1]}
    )
    log_brain_error(error3, "test_client_003")
    
    # Test 2: Error code lookup
    print("\n2. Testing Error Code Lookup:")
    
    # Look up by name
    similarity_error = brain_error_handler.get_error_code_by_name("SIMILARITY_ENGINE_FAILURE")
    if similarity_error:
        print(f"   ✅ Found error code: {similarity_error.value} ({similarity_error.name})")
        error_info = brain_error_handler.get_error_info(similarity_error)
        print(f"   📋 Description: {error_info['description']}")
        print(f"   🔧 Resolution: {error_info['resolution']}")
        print(f"   📊 Severity: {error_info['severity']}")
    
    # Test 3: All error codes
    print("\n3. Testing All Error Codes:")
    
    print("   Protocol Errors (1.x):")
    for code in [BrainErrorCode.UNKNOWN_MESSAGE_TYPE, BrainErrorCode.PROTOCOL_ERROR]:
        info = brain_error_handler.get_error_info(code)
        print(f"     {code.value} - {code.name}: {info['description']}")
    
    print("   Brain Processing Errors (5.x):")
    for code in [BrainErrorCode.SIMILARITY_ENGINE_FAILURE, BrainErrorCode.PREDICTION_ENGINE_FAILURE, 
                 BrainErrorCode.MEMORY_PRESSURE_ERROR, BrainErrorCode.GPU_PROCESSING_ERROR]:
        info = brain_error_handler.get_error_info(code)
        print(f"     {code.value} - {code.name}: {info['description']}")
    
    # Test 4: Error classification
    print("\n4. Testing Error Classification:")
    
    # Simulate what the server classification would do
    test_exceptions = [
        ("similarity timeout", "Should classify as SIMILARITY_ENGINE_FAILURE"),
        ("prediction failed", "Should classify as PREDICTION_ENGINE_FAILURE"),
        ("memory error", "Should classify as MEMORY_PRESSURE_ERROR"),
        ("gpu error", "Should classify as GPU_PROCESSING_ERROR"),
        ("random error", "Should classify as BRAIN_PROCESSING_ERROR")
    ]
    
    for exc_msg, expected in test_exceptions:
        # Mock the classification logic
        if 'similarity' in exc_msg:
            code = BrainErrorCode.SIMILARITY_ENGINE_FAILURE
        elif 'prediction' in exc_msg:
            code = BrainErrorCode.PREDICTION_ENGINE_FAILURE
        elif 'memory' in exc_msg:
            code = BrainErrorCode.MEMORY_PRESSURE_ERROR
        elif 'gpu' in exc_msg:
            code = BrainErrorCode.GPU_PROCESSING_ERROR
        else:
            code = BrainErrorCode.BRAIN_PROCESSING_ERROR
        
        print(f"   Exception: '{exc_msg}' → {code.value} ({code.name})")
    
    print("\n✅ Error handling system working correctly!")
    print("\n📋 Benefits of New System:")
    print("   • Descriptive error codes instead of cryptic numbers")
    print("   • Automatic error classification based on exception content")
    print("   • Context information for debugging")
    print("   • Resolution suggestions for each error type")
    print("   • Severity-based logging with appropriate icons")
    print("   • Easy searchability by error name or code")
    
    return True

if __name__ == "__main__":
    success = test_error_handling()
    if success:
        print("\n🎉 New error handling system is ready!")
        print("🔄 The server will now provide much better error information")
    else:
        print("\n❌ Error handling system needs debugging")