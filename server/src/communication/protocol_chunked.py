"""
Chunked protocol for handling large messages with small TCP buffers.

Since system TCP buffers are limited to ~212KB but we need to send 1.2MB messages,
we need to handle chunked transmission properly.
"""

import struct
from typing import List, Tuple
import socket


class ChunkedProtocol:
    """
    Protocol that handles large messages by reading them in chunks.
    Works with limited TCP buffer sizes.
    """
    
    # Message types
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    # Magic bytes for message start
    MAGIC_BYTES = 0xDEADBEEF
    
    def __init__(self, max_vector_size: int = 10_000_000):
        self.max_vector_size = max_vector_size
        self.chunk_size = 65536  # 64KB chunks for reading
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """
        Receive message handling large data properly.
        """
        sock.settimeout(timeout)
        
        try:
            # Read magic bytes
            magic_bytes = self._receive_exactly(sock, 4)
            magic = struct.unpack('!I', magic_bytes)[0]
            
            if magic != self.MAGIC_BYTES:
                # Check if it's zeros (connection issue)
                if magic == 0x00000000:
                    raise ConnectionError("Received null bytes - client may have disconnected")
                raise ValueError(f"Invalid magic bytes: 0x{magic:08X}")
            
            # Read message length
            length_bytes = self._receive_exactly(sock, 4)
            message_length = struct.unpack('!I', length_bytes)[0]
            
            # Sanity check
            if message_length > (self.max_vector_size * 4 + 5):
                raise ValueError(f"Message too large: {message_length} bytes")
            
            # Read message type
            type_byte = self._receive_exactly(sock, 1)
            msg_type = type_byte[0]
            
            # Read vector length
            vec_len_bytes = self._receive_exactly(sock, 4)
            vector_length = struct.unpack('!I', vec_len_bytes)[0]
            
            if vector_length > self.max_vector_size:
                raise ValueError(f"Vector too large: {vector_length}")
            
            # Read vector data in chunks (this is the key part!)
            vector_bytes = vector_length * 4
            vector_data = b''
            
            # Read in chunks to handle large messages
            while len(vector_data) < vector_bytes:
                chunk_size = min(self.chunk_size, vector_bytes - len(vector_data))
                chunk = self._receive_exactly(sock, chunk_size)
                vector_data += chunk
                
                # Progress for large messages
                if vector_bytes > 1000000:  # 1MB+
                    progress = len(vector_data) / vector_bytes * 100
                    if progress % 10 < 0.1:  # Log every 10%
                        print(f"   Receiving large message: {progress:.0f}%")
            
            # Unpack vector
            vector = list(struct.unpack(f'{vector_length}f', vector_data))
            
            return msg_type, vector
            
        except socket.timeout:
            raise TimeoutError("Message receive timeout")
        except Exception as e:
            # Re-raise with more context
            raise ConnectionError(f"Protocol error: {e}")
    
    def encode_handshake(self, capabilities: List[float]) -> bytes:
        """Encode handshake message."""
        return self._encode_vector(capabilities, self.MSG_HANDSHAKE)
    
    def encode_action_output(self, action_vector: List[float]) -> bytes:
        """Encode motor commands."""
        return self._encode_vector(action_vector, self.MSG_ACTION_OUTPUT)
    
    def encode_error(self, error_code: float = 0.0) -> bytes:
        """Encode error message."""
        return self._encode_vector([error_code], self.MSG_ERROR)
    
    def _encode_vector(self, vector: List[float], msg_type: int) -> bytes:
        """Encode vector with proper header."""
        if len(vector) > self.max_vector_size:
            raise ValueError(f"Vector too large: {len(vector)}")
        
        # Pack vector data
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Build message
        message_length = 1 + 4 + len(vector_data)  # type + vec_len + data
        header = struct.pack('!IIBI', self.MAGIC_BYTES, message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def send_message(self, sock: socket.socket, message_data: bytes) -> bool:
        """
        Send message in chunks to handle limited buffers.
        """
        try:
            total_size = len(message_data)
            sent = 0
            
            # Send in chunks
            while sent < total_size:
                chunk_size = min(self.chunk_size, total_size - sent)
                chunk = message_data[sent:sent + chunk_size]
                
                # Send chunk (may still be partial)
                bytes_sent = sock.send(chunk)
                if bytes_sent == 0:
                    raise ConnectionError("Socket connection broken")
                sent += bytes_sent
                
                # Progress for large messages
                if total_size > 1000000:  # 1MB+
                    progress = sent / total_size * 100
                    if progress % 10 < 0.1:  # Log every 10%
                        print(f"   Sending large message: {progress:.0f}%")
            
            return True
            
        except (socket.error, BrokenPipeError) as e:
            print(f"   Send error: {e}")
            return False
    
    def _receive_exactly(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(min(num_bytes - len(data), self.chunk_size))
            if not chunk:
                if len(data) == 0:
                    raise ConnectionError("Socket closed - no data received")
                else:
                    raise ConnectionError(f"Socket closed after {len(data)} bytes (expected {num_bytes})")
            data += chunk
        return data