"""
Minimal Protocol

Embarrassingly simple TCP protocol for brain-robot communication.
Raw vector exchange - no JSON, no complex packets, just numbers.
"""

import struct
from typing import List, Tuple, Optional
import socket


class MessageProtocol:
    """
    Ultra-simple message protocol for brain-robot communication.
    
    Message Format:
    1. Message length (4 bytes, uint32)
    2. Message type (1 byte): 0=sensory_input, 1=action_output, 2=handshake
    3. Vector length (4 bytes, uint32) 
    4. Vector data (vector_length * 4 bytes, float32 each)
    
    Total overhead: 9 bytes per message
    """
    
    # Message types
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1  
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    def __init__(self, max_vector_size: int = None):
        # Support up to 4K resolution (3840×2160 = 8,294,400 pixels)
        # Plus some overhead for other sensors
        # This allows ~32MB messages (8.3M * 4 bytes)
        # Can be overridden for specific use cases
        if max_vector_size is None:
            max_vector_size = 10_000_000  # 10 million values (~40MB max)
        self.max_vector_size = max_vector_size
    
    def encode_sensory_input(self, sensory_vector: List[float]) -> bytes:
        """Encode sensory input vector for transmission."""
        return self._encode_vector(sensory_vector, self.MSG_SENSORY_INPUT)
    
    def encode_action_output(self, action_vector: List[float]) -> bytes:
        """Encode action output vector for transmission."""
        return self._encode_vector(action_vector, self.MSG_ACTION_OUTPUT)
    
    def encode_handshake(self, capabilities: List[float]) -> bytes:
        """Encode handshake message with robot capabilities."""
        return self._encode_vector(capabilities, self.MSG_HANDSHAKE)
    
    def encode_error(self, error_code: float = 0.0) -> bytes:
        """Encode error message."""
        return self._encode_vector([error_code], self.MSG_ERROR)
    
    def _encode_vector(self, vector: List[float], msg_type: int) -> bytes:
        """Encode a vector with message header."""
        if len(vector) > self.max_vector_size:
            raise ValueError(f"Vector too large: {len(vector)} > {self.max_vector_size}")
        
        # Convert to float32 for consistent precision
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Message length = type(1) + vector_length(4) + vector_data
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: length + type + vector_length + vector_data
        header = struct.pack('!IBI', message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def decode_message(self, data: bytes) -> Tuple[int, List[float]]:
        """
        Decode received message.
        
        Returns:
            Tuple of (message_type, vector_data)
        """
        if len(data) < 9:  # Minimum message size
            raise ValueError("Message too short")
        
        # Unpack header
        message_length, msg_type, vector_length = struct.unpack('!IBI', data[:9])
        
        # Validate message
        expected_length = 1 + 4 + (vector_length * 4)
        if message_length != expected_length:
            raise ValueError(f"Invalid message length: {message_length} != {expected_length}")
        
        if vector_length > self.max_vector_size:
            raise ValueError(f"Vector too large: {vector_length}")
        
        # Extract vector data
        vector_data = data[9:9 + (vector_length * 4)]
        if len(vector_data) != vector_length * 4:
            raise ValueError("Incomplete vector data")
        
        # Unpack vector
        vector = list(struct.unpack(f'{vector_length}f', vector_data))
        
        return msg_type, vector
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """
        Receive and decode a complete message from socket.
        
        Args:
            sock: Socket to receive from
            timeout: Receive timeout in seconds
            
        Returns:
            Tuple of (message_type, vector_data)
        """
        sock.settimeout(timeout)
        
        try:
            # Receive message length first
            length_data = self._receive_exactly(sock, 4)
            message_length = struct.unpack('!I', length_data)[0]
            
            if message_length > (self.max_vector_size * 4 + 5):  # Sanity check
                raise ValueError(f"Message too large: {message_length}")
            
            # Receive the rest of the message
            message_data = self._receive_exactly(sock, message_length)
            
            # Decode complete message
            return self.decode_message(length_data + message_data)
            
        except socket.timeout:
            raise TimeoutError("Message receive timeout")
    
    def send_message(self, sock: socket.socket, message_data: bytes) -> bool:
        """
        Send complete message over socket.
        
        Args:
            sock: Socket to send to
            message_data: Encoded message bytes
            
        Returns:
            True if sent successfully
        """
        try:
            total_sent = 0
            while total_sent < len(message_data):
                sent = sock.send(message_data[total_sent:])
                if sent == 0:
                    return False
                total_sent += sent
            return True
            
        except (socket.error, BrokenPipeError):
            return False
    
    def _receive_exactly(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly")
            data += chunk
        return data
    
    def get_message_info(self, data: bytes) -> dict:
        """Get information about a message without fully decoding."""
        if len(data) < 9:
            return {'error': 'Message too short'}
        
        try:
            message_length, msg_type, vector_length = struct.unpack('!IBI', data[:9])
            
            msg_type_names = {
                self.MSG_SENSORY_INPUT: 'sensory_input',
                self.MSG_ACTION_OUTPUT: 'action_output', 
                self.MSG_HANDSHAKE: 'handshake',
                self.MSG_ERROR: 'error'
            }
            
            return {
                'message_length': message_length,
                'message_type': msg_type,
                'message_type_name': msg_type_names.get(msg_type, 'unknown'),
                'vector_length': vector_length,
                'total_bytes': message_length + 4  # +4 for length header
            }
            
        except struct.error:
            return {'error': 'Invalid message format'}


class MessageProtocolError(Exception):
    """Exception raised for protocol-related errors."""
    pass