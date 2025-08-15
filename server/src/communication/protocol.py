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
    1. Magic bytes (4 bytes): 0xDEADBEEF for corruption detection
    2. Message length (4 bytes, uint32) - excludes magic bytes
    3. Message type (1 byte): 0=sensory_input, 1=action_output, 2=handshake
    4. Vector length (4 bytes, uint32) 
    5. Vector data (vector_length * 4 bytes, float32 each)
    
    Total overhead: 13 bytes per message
    """
    
    # Message types
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1  
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    # Protocol constants
    MAGIC_BYTES = 0xDEADBEEF
    
    # Validation limits
    MIN_VECTOR_LENGTH = 0
    MAX_REASONABLE_VECTOR_LENGTH = 100_000  # 100K elements for safety
    MAX_MESSAGE_SIZE_BYTES = 50_000_000     # 50MB absolute maximum
    
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
        
        # Message length = type(1) + vector_length(4) + vector_data (excludes magic)
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: magic + length + type + vector_length + vector_data
        header = struct.pack('!IIBI', self.MAGIC_BYTES, message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def decode_message(self, data: bytes) -> Tuple[int, List[float]]:
        """
        Decode received message.
        
        Returns:
            Tuple of (message_type, vector_data)
        """
        if len(data) < 13:  # Minimum message size with magic bytes
            raise ValueError("Message too short")
        
        # Unpack header with magic bytes
        magic, message_length, msg_type, vector_length = struct.unpack('!IIBI', data[:13])
        
        # Validate magic bytes
        if magic != self.MAGIC_BYTES:
            raise MessageProtocolError(f"Invalid magic bytes: 0x{magic:08X} != 0x{self.MAGIC_BYTES:08X}")
        
        # Validate message length consistency
        expected_length = 1 + 4 + (vector_length * 4)
        if message_length != expected_length:
            raise MessageProtocolError(f"Message length mismatch: {message_length} != {expected_length} (type+vec_len+data)")
        
        # Additional sanity checks
        if msg_type not in [self.MSG_SENSORY_INPUT, self.MSG_ACTION_OUTPUT, self.MSG_HANDSHAKE, self.MSG_ERROR]:
            raise MessageProtocolError(f"Unknown message type: {msg_type}")
        
        # Enhanced vector validation
        if vector_length < self.MIN_VECTOR_LENGTH:
            raise MessageProtocolError(f"Vector length negative: {vector_length}")
        if vector_length > self.MAX_REASONABLE_VECTOR_LENGTH:
            raise MessageProtocolError(f"Vector length unreasonable: {vector_length} > {self.MAX_REASONABLE_VECTOR_LENGTH}")
        if vector_length > self.max_vector_size:
            raise MessageProtocolError(f"Vector too large: {vector_length} > {self.max_vector_size}")
        
        # Extract vector data
        vector_data = data[13:13 + (vector_length * 4)]
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
            # Receive magic bytes and message length first
            header_data = self._receive_exactly(sock, 8)  # magic + length
            magic, message_length = struct.unpack('!II', header_data)
            
            # Validate magic bytes first
            if magic != self.MAGIC_BYTES:
                # Try to resynchronize stream
                self._resync_stream(sock, timeout)
                raise MessageProtocolError(f"Invalid magic bytes: 0x{magic:08X}, stream may be corrupt")
            
            # Validate message length with multiple checks
            if message_length < 5:  # Minimum: type(1) + vector_length(4)
                raise MessageProtocolError(f"Message too small: {message_length} bytes")
            if message_length > self.MAX_MESSAGE_SIZE_BYTES:
                raise MessageProtocolError(f"Message too large: {message_length} bytes > {self.MAX_MESSAGE_SIZE_BYTES}")
            if message_length > (self.max_vector_size * 4 + 5):
                raise MessageProtocolError(f"Message larger than max vector: {message_length} bytes")
            
            # Receive the rest of the message
            message_data = self._receive_exactly(sock, message_length)
            
            # Decode complete message
            return self.decode_message(header_data + message_data)
            
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
    
    def _resync_stream(self, sock: socket.socket, timeout: float):
        """
        Attempt to resynchronize the stream by looking for magic bytes.
        
        This is called when we detect stream corruption to try to recover.
        """
        sock.settimeout(min(timeout, 2.0))  # Shorter timeout for resync
        
        buffer = b''
        max_search_bytes = 1024  # Don't search forever
        searched = 0
        
        try:
            while searched < max_search_bytes:
                # Read one byte at a time looking for magic
                byte = sock.recv(1)
                if not byte:
                    raise ConnectionError("Connection closed during resync")
                
                buffer += byte
                searched += 1
                
                # Keep only last 4 bytes for magic detection
                if len(buffer) > 4:
                    buffer = buffer[-4:]
                
                # Check if we found magic bytes
                if len(buffer) == 4:
                    potential_magic = struct.unpack('!I', buffer)[0]
                    if potential_magic == self.MAGIC_BYTES:
                        # Found magic bytes! Stream should be synchronized now
                        return
                        
        except socket.timeout:
            pass  # Give up on resync
        
        # Could not resynchronize
        raise MessageProtocolError("Could not resynchronize stream")
    
    def get_message_info(self, data: bytes) -> dict:
        """Get information about a message without fully decoding."""
        if len(data) < 13:
            return {'error': 'Message too short'}
        
        try:
            magic, message_length, msg_type, vector_length = struct.unpack('!IIBI', data[:13])
            
            msg_type_names = {
                self.MSG_SENSORY_INPUT: 'sensory_input',
                self.MSG_ACTION_OUTPUT: 'action_output', 
                self.MSG_HANDSHAKE: 'handshake',
                self.MSG_ERROR: 'error'
            }
            
            return {
                'magic_bytes': f'0x{magic:08X}',
                'magic_valid': magic == self.MAGIC_BYTES,
                'message_length': message_length,
                'message_type': msg_type,
                'message_type_name': msg_type_names.get(msg_type, 'unknown'),
                'vector_length': vector_length,
                'total_bytes': message_length + 8  # +8 for magic + length headers
            }
            
        except struct.error:
            return {'error': 'Invalid message format'}


class MessageProtocolError(Exception):
    """Exception raised for protocol-related errors."""
    pass