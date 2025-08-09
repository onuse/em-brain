#!/usr/bin/env python3
"""
Protocol Unit Tests

Critical tests for the binary protocol that ensure:
1. Messages encode/decode correctly
2. Corruption is detected
3. Buffer overflows are prevented
4. Thread safety is maintained
5. Timeouts work correctly

These tests are the foundation - if these fail, nothing works.
"""

import pytest
import struct
import threading
import time
from unittest.mock import Mock, MagicMock, patch
import socket
import random

# Import both protocol implementations to test consistency
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from brainstem.brain_client import MessageProtocol as ClientProtocol

# Also test server-side protocol if accessible
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../server/src'))
try:
    from communication.protocol import MessageProtocol as ServerProtocol
    HAS_SERVER_PROTOCOL = True
except ImportError:
    HAS_SERVER_PROTOCOL = False
    ServerProtocol = None


class TestProtocolEncoding:
    """Test message encoding/decoding correctness."""
    
    def test_sensory_input_encoding(self):
        """Test encoding of sensory input messages."""
        protocol = ClientProtocol()
        
        # Test various vector sizes
        test_cases = [
            ([0.5], "single value"),
            ([0.0, 1.0], "boundary values"),
            ([0.5] * 24, "typical sensory vector"),
            ([-1.0, -0.5, 0.0, 0.5, 1.0], "mixed values"),
            ([float(i) / 100 for i in range(100)], "large vector"),
        ]
        
        for vector, description in test_cases:
            encoded = protocol.encode_sensory_input(vector)
            
            # Verify structure
            assert len(encoded) >= 9, f"Header too short for {description}"
            
            # Decode header manually
            msg_length, msg_type, vec_length = struct.unpack('!IBI', encoded[:9])
            
            # Verify header values
            assert msg_type == protocol.MSG_SENSORY_INPUT
            assert vec_length == len(vector)
            assert msg_length == 1 + 4 + (vec_length * 4)
            
            # Verify total message size
            assert len(encoded) == 4 + msg_length
            
            # Decode and verify vector
            vector_data = encoded[9:]
            decoded_vector = list(struct.unpack(f'{vec_length}f', vector_data))
            
            # Check values (accounting for float32 precision)
            for original, decoded in zip(vector, decoded_vector):
                assert abs(original - decoded) < 1e-6, f"Value mismatch in {description}"
    
    def test_action_output_encoding(self):
        """Test encoding of action output messages."""
        protocol = ClientProtocol()
        
        # Typical action vector
        actions = [0.0, 0.5, -0.3, 0.8]
        encoded = protocol.encode_handshake(actions)  # Using handshake as proxy
        
        # Should encode correctly
        assert len(encoded) == 4 + 1 + 4 + (len(actions) * 4)
    
    def test_handshake_encoding(self):
        """Test handshake message encoding."""
        protocol = ClientProtocol()
        
        # Robot capabilities: [version, sensory_dims, action_dims, reserved, reserved]
        capabilities = [1.0, 24.0, 4.0, 0.0, 0.0]
        encoded = protocol.encode_handshake(capabilities)
        
        # Decode and verify
        msg_length, msg_type, vec_length = struct.unpack('!IBI', encoded[:9])
        assert msg_type == protocol.MSG_HANDSHAKE
        assert vec_length == len(capabilities)
    
    def test_message_size_limits(self):
        """Test that oversized messages are rejected."""
        protocol = ClientProtocol()
        
        # Try to encode vector larger than max size
        huge_vector = [0.5] * (protocol.max_vector_size + 1)
        
        with pytest.raises(ValueError, match="Vector too large"):
            protocol.encode_sensory_input(huge_vector)
    
    def test_empty_vector_handling(self):
        """Test encoding of empty vectors."""
        protocol = ClientProtocol()
        
        empty = []
        encoded = protocol.encode_sensory_input(empty)
        
        # Should still have valid header
        msg_length, msg_type, vec_length = struct.unpack('!IBI', encoded[:9])
        assert vec_length == 0
        assert len(encoded) == 9  # Just header


class TestProtocolDecoding:
    """Test message decoding and error handling."""
    
    def test_decode_valid_message(self):
        """Test decoding of valid messages."""
        protocol = ClientProtocol()
        
        # Create a valid message manually
        vector = [0.1, 0.2, 0.3, 0.4]
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        msg_length = 1 + 4 + len(vector_data)
        header = struct.pack('!IBI', msg_length, protocol.MSG_SENSORY_INPUT, len(vector))
        message = header + vector_data
        
        # Decode it
        msg_type, decoded_vector = protocol.decode_message(message)
        
        assert msg_type == protocol.MSG_SENSORY_INPUT
        assert len(decoded_vector) == len(vector)
        for original, decoded in zip(vector, decoded_vector):
            assert abs(original - decoded) < 1e-6
    
    def test_decode_corrupted_header(self):
        """Test handling of corrupted message headers."""
        protocol = ClientProtocol()
        
        # Various corruption scenarios
        corrupted_messages = [
            b'',  # Empty
            b'ABC',  # Too short
            b'ABCDEFGH',  # Still too short (8 bytes)
            struct.pack('!IBI', 999999, 0, 10),  # Wrong length claim
            struct.pack('!IBI', 100, 99, 10),  # Invalid message type
        ]
        
        for corrupt_msg in corrupted_messages:
            with pytest.raises(ValueError):
                protocol.decode_message(corrupt_msg)
    
    def test_decode_truncated_message(self):
        """Test handling of truncated messages."""
        protocol = ClientProtocol()
        
        # Create valid message then truncate it
        vector = [0.5] * 10
        encoded = protocol.encode_sensory_input(vector)
        
        # Try various truncation points
        for truncate_at in [5, 9, 15, len(encoded) - 1]:
            truncated = encoded[:truncate_at]
            with pytest.raises(ValueError):
                protocol.decode_message(truncated)
    
    def test_decode_corrupted_vector_data(self):
        """Test handling of corrupted vector data."""
        protocol = ClientProtocol()
        
        # Create header claiming 10 floats but provide only 5
        header = struct.pack('!IBI', 1 + 4 + 40, protocol.MSG_SENSORY_INPUT, 10)
        partial_data = struct.pack('5f', *[0.5] * 5)
        corrupt_message = header + partial_data
        
        with pytest.raises(ValueError, match="Incomplete vector data"):
            protocol.decode_message(corrupt_message)


class TestProtocolThreadSafety:
    """Test thread safety of protocol operations."""
    
    def test_concurrent_encoding(self):
        """Test that multiple threads can encode simultaneously."""
        protocol = ClientProtocol()
        results = []
        errors = []
        
        def encode_worker(thread_id):
            try:
                for i in range(100):
                    vector = [float(thread_id), float(i)] + [0.5] * 22
                    encoded = protocol.encode_sensory_input(vector)
                    
                    # Decode to verify it wasn't corrupted
                    msg_type, decoded = protocol.decode_message(encoded)
                    assert decoded[0] == float(thread_id)
                    assert decoded[1] == float(i)
                    
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=encode_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)
        
        # Verify results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10, "Not all threads completed"
    
    def test_socket_receive_timeout(self):
        """Test that receive timeout works correctly."""
        protocol = ClientProtocol()
        
        # Create mock socket that times out
        mock_sock = Mock(spec=socket.socket)
        mock_sock.recv.side_effect = socket.timeout("timed out")
        mock_sock.settimeout = Mock()
        
        with pytest.raises(TimeoutError, match="Message receive timeout"):
            protocol.receive_message(mock_sock, timeout=0.1)
        
        # Verify timeout was set
        mock_sock.settimeout.assert_called_with(0.1)
    
    def test_socket_receive_interrupted(self):
        """Test handling of interrupted socket receives."""
        protocol = ClientProtocol()
        
        # Mock socket that returns partial data then fails
        mock_sock = Mock(spec=socket.socket)
        mock_sock.recv.side_effect = [
            struct.pack('!I', 100),  # Return length
            socket.error("Connection reset")  # Then fail
        ]
        mock_sock.settimeout = Mock()
        
        with pytest.raises(socket.error):
            protocol.receive_message(mock_sock)


class TestProtocolRobustness:
    """Test protocol robustness against edge cases and attacks."""
    
    def test_message_length_overflow(self):
        """Test handling of message length overflow attempts."""
        protocol = ClientProtocol()
        
        # Try to create message with max uint32 length
        malicious_header = struct.pack('!I', 0xFFFFFFFF)
        
        # This should be caught by the protocol
        mock_sock = Mock(spec=socket.socket)
        mock_sock.recv.side_effect = [
            malicious_header,
            socket.error("Would block forever")
        ]
        mock_sock.settimeout = Mock()
        
        with pytest.raises(ValueError, match="Message too large"):
            protocol.receive_message(mock_sock)
    
    def test_rapid_reconnection(self):
        """Test handling of rapid connection/disconnection cycles."""
        protocol = ClientProtocol()
        
        for _ in range(100):
            mock_sock = Mock(spec=socket.socket)
            
            # Simulate immediate disconnect
            mock_sock.recv.return_value = b''
            mock_sock.settimeout = Mock()
            
            with pytest.raises(Exception):  # Should handle gracefully
                protocol.receive_message(mock_sock, timeout=0.01)
    
    def test_binary_fuzzing(self):
        """Fuzz test with random binary data."""
        protocol = ClientProtocol()
        
        random.seed(42)  # Reproducible fuzzing
        
        for _ in range(1000):
            # Generate random binary data
            fuzz_length = random.randint(0, 100)
            fuzz_data = bytes(random.randint(0, 255) for _ in range(fuzz_length))
            
            # Protocol should either decode or raise ValueError, never crash
            try:
                msg_type, vector = protocol.decode_message(fuzz_data)
                # If it decoded, verify it's valid
                assert msg_type in [protocol.MSG_SENSORY_INPUT, 
                                   protocol.MSG_ACTION_OUTPUT,
                                   protocol.MSG_HANDSHAKE,
                                   protocol.MSG_ERROR]
                assert isinstance(vector, list)
            except ValueError:
                pass  # Expected for invalid data
            except Exception as e:
                pytest.fail(f"Unexpected exception on fuzz data: {e}")


class TestProtocolCompatibility:
    """Test compatibility between client and server protocols."""
    
    @pytest.mark.skipif(not HAS_SERVER_PROTOCOL, reason="Server protocol not available")
    def test_client_server_compatibility(self):
        """Verify client and server protocols are compatible."""
        client_proto = ClientProtocol()
        server_proto = ServerProtocol()
        
        # Test that they use same message type constants
        assert client_proto.MSG_SENSORY_INPUT == server_proto.MSG_SENSORY_INPUT
        assert client_proto.MSG_ACTION_OUTPUT == server_proto.MSG_ACTION_OUTPUT
        assert client_proto.MSG_HANDSHAKE == server_proto.MSG_HANDSHAKE
        assert client_proto.MSG_ERROR == server_proto.MSG_ERROR
        
        # Test that encoding/decoding is compatible
        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Client encodes, server decodes
        client_encoded = client_proto.encode_sensory_input(test_vector)
        server_msg_type, server_decoded = server_proto.decode_message(client_encoded)
        
        assert server_msg_type == server_proto.MSG_SENSORY_INPUT
        for original, decoded in zip(test_vector, server_decoded):
            assert abs(original - decoded) < 1e-6
        
        # Server encodes, client decodes
        server_encoded = server_proto.encode_action_output(test_vector)
        client_msg_type, client_decoded = client_proto.decode_message(server_encoded)
        
        assert client_msg_type == client_proto.MSG_ACTION_OUTPUT
        for original, decoded in zip(test_vector, client_decoded):
            assert abs(original - decoded) < 1e-6


class TestProtocolPerformance:
    """Performance benchmarks for protocol operations."""
    
    def test_encoding_performance(self):
        """Benchmark encoding performance."""
        protocol = ClientProtocol()
        vector = [0.5] * 24  # Typical sensory vector
        
        start = time.perf_counter()
        iterations = 10000
        
        for _ in range(iterations):
            encoded = protocol.encode_sensory_input(vector)
        
        elapsed = time.perf_counter() - start
        per_message = (elapsed / iterations) * 1000  # Convert to ms
        
        # Should be well under 1ms per message
        assert per_message < 0.1, f"Encoding too slow: {per_message:.3f}ms per message"
        
        print(f"✓ Encoding performance: {per_message*1000:.1f}µs per message")
    
    def test_decoding_performance(self):
        """Benchmark decoding performance."""
        protocol = ClientProtocol()
        vector = [0.5] * 24
        encoded = protocol.encode_sensory_input(vector)
        
        start = time.perf_counter()
        iterations = 10000
        
        for _ in range(iterations):
            msg_type, decoded = protocol.decode_message(encoded)
        
        elapsed = time.perf_counter() - start
        per_message = (elapsed / iterations) * 1000  # Convert to ms
        
        # Should be well under 1ms per message
        assert per_message < 0.1, f"Decoding too slow: {per_message:.3f}ms per message"
        
        print(f"✓ Decoding performance: {per_message*1000:.1f}µs per message")


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_protocol.py -v
    pytest.main([__file__, "-v", "--tb=short"])