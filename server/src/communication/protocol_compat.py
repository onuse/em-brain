"""
Backward-compatible protocol that can handle both old and new message formats.
This allows the server to work with robots that haven't been updated yet.
"""

import struct
from typing import List, Tuple, Optional
import socket


class BackwardCompatibleProtocol:
    """
    Protocol that can handle both:
    - OLD format: [length][type][vector_length][data]
    - NEW format: [magic][length][type][vector_length][data]
    """
    
    # Message types
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    # Magic bytes for new protocol
    MAGIC_BYTES = 0xDEADBEEF
    
    def __init__(self, max_vector_size: int = 10_000_000):
        self.max_vector_size = max_vector_size
        self.client_uses_magic = None  # Will be detected on first message
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """
        Receive message in either old or new format.
        Auto-detects which format the client is using.
        """
        sock.settimeout(timeout)
        
        try:
            # Read first 4 bytes
            first_bytes = self._receive_exactly(sock, 4)
            first_word = struct.unpack('!I', first_bytes)[0]
            
            # Check if it's magic bytes (new protocol) or message length (old protocol)
            if first_word == self.MAGIC_BYTES:
                # NEW PROTOCOL with magic bytes
                if self.client_uses_magic is None:
                    print("   Client using NEW protocol (with magic bytes)")
                    self.client_uses_magic = True
                
                # Read actual message length
                length_bytes = self._receive_exactly(sock, 4)
                message_length = struct.unpack('!I', length_bytes)[0]
                
                # Validate length
                if message_length > (self.max_vector_size * 4 + 5):
                    raise ValueError(f"Message too large: {message_length}")
                
                # Read rest of message
                message_data = self._receive_exactly(sock, message_length)
                
                # Parse message
                msg_type = message_data[0]
                vector_length = struct.unpack('!I', message_data[1:5])[0]
                
                # Validate vector length (support vision data)
                if vector_length > self.max_vector_size:
                    raise ValueError(f"Vector too large: {vector_length}")
                
                # Extract vector
                vector_data = message_data[5:5 + (vector_length * 4)]
                vector = list(struct.unpack(f'{vector_length}f', vector_data))
                
                return msg_type, vector
                
            else:
                # OLD PROTOCOL without magic bytes
                # first_word is the message length
                if self.client_uses_magic is None:
                    print("   Client using OLD protocol (no magic bytes)")
                    self.client_uses_magic = False
                
                message_length = first_word
                
                # Validate length - OLD protocol messages for 307,212 values would be:
                # 1 (type) + 4 (vector_length) + 307212*4 (data) = 1,228,853 bytes
                MAX_OLD_MESSAGE = 1_500_000  # 1.5MB reasonable for 307K values
                
                if message_length > MAX_OLD_MESSAGE:
                    # This is likely corrupt data
                    raise ValueError(f"Message too large: {message_length} bytes (max {MAX_OLD_MESSAGE})")
                
                if message_length > (self.max_vector_size * 4 + 5):
                    raise ValueError(f"Message too large: {message_length}")
                
                # Read rest of message
                message_data = self._receive_exactly(sock, message_length)
                
                # Parse old format message
                msg_type = message_data[0]
                vector_length = struct.unpack('!I', message_data[1:5])[0]
                
                # Validate vector length
                if vector_length > self.max_vector_size:
                    raise ValueError(f"Vector too large: {vector_length}")
                
                # Extract vector
                vector_data = message_data[5:5 + (vector_length * 4)]
                vector = list(struct.unpack(f'{vector_length}f', vector_data))
                
                return msg_type, vector
                
        except socket.timeout:
            raise TimeoutError("Message receive timeout")
    
    def send_message(self, sock: socket.socket, message_data: bytes) -> bool:
        """Send message (uses format client expects)."""
        try:
            sock.sendall(message_data)
            return True
        except (socket.error, BrokenPipeError):
            return False
    
    def encode_handshake(self, capabilities: List[float]) -> bytes:
        """Encode handshake (uses client's expected format)."""
        return self._encode_vector(capabilities, self.MSG_HANDSHAKE)
    
    def encode_action_output(self, action_vector: List[float]) -> bytes:
        """Encode action output (uses client's expected format)."""
        return self._encode_vector(action_vector, self.MSG_ACTION_OUTPUT)
    
    def encode_error(self, error_code: float = 0.0) -> bytes:
        """Encode error message."""
        return self._encode_vector([error_code], self.MSG_ERROR)
    
    def _encode_vector(self, vector: List[float], msg_type: int) -> bytes:
        """
        Encode vector in the format the client expects.
        If we haven't received a message yet, use new format.
        """
        if len(vector) > self.max_vector_size:
            raise ValueError(f"Vector too large: {len(vector)} > {self.max_vector_size}")
        
        # Convert to float32
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Decide which format to use
        use_magic = self.client_uses_magic if self.client_uses_magic is not None else True
        
        if use_magic:
            # NEW format with magic bytes
            message_length = 1 + 4 + len(vector_data)
            header = struct.pack('!IIBI', self.MAGIC_BYTES, message_length, msg_type, len(vector))
            return header + vector_data
        else:
            # OLD format without magic bytes
            message_length = 1 + 4 + len(vector_data)
            header = struct.pack('!IBI', message_length, msg_type, len(vector))
            return header + vector_data
    
    def _receive_exactly(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly")
            data += chunk
        return data