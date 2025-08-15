#!/usr/bin/env python3
"""
Test that both client and server are using the same protocol version with magic bytes.
"""

import sys
import struct

def check_protocol_versions():
    """Check that both sides have matching protocol implementations."""
    
    print("Protocol Version Check")
    print("=" * 60)
    
    # Check server protocol
    print("\n1. Server Protocol (server/src/communication/protocol.py):")
    try:
        from server.src.communication.protocol import MessageProtocol as ServerProtocol
        server_proto = ServerProtocol()
        
        # Check for magic bytes constant
        if hasattr(ServerProtocol, 'MAGIC_BYTES'):
            print(f"   ✅ Has MAGIC_BYTES: 0x{ServerProtocol.MAGIC_BYTES:08X}")
        else:
            print("   ❌ Missing MAGIC_BYTES constant")
            
        # Check max vector size
        if hasattr(ServerProtocol, 'MAX_REASONABLE_VECTOR_LENGTH'):
            print(f"   ✅ MAX_REASONABLE_VECTOR_LENGTH: {ServerProtocol.MAX_REASONABLE_VECTOR_LENGTH:,}")
        else:
            print("   ❌ Missing MAX_REASONABLE_VECTOR_LENGTH")
            
        # Try encoding a test message
        test_vec = [1.0, 2.0, 3.0]
        encoded = server_proto.encode_handshake(test_vec)
        
        # Check if it starts with magic bytes
        if len(encoded) >= 4:
            magic = struct.unpack('!I', encoded[:4])[0]
            if magic == 0xDEADBEEF:
                print(f"   ✅ Encodes with correct magic bytes")
            else:
                print(f"   ❌ Wrong magic bytes in encoding: 0x{magic:08X}")
        else:
            print("   ❌ Encoded message too short")
            
    except ImportError as e:
        print(f"   ❌ Could not import server protocol: {e}")
    except Exception as e:
        print(f"   ❌ Error testing server protocol: {e}")
    
    # Check client protocol
    print("\n2. Client Protocol (client_picarx/src/brainstem/brain_client.py):")
    try:
        sys.path.insert(0, 'client_picarx/src/brainstem')
        from brain_client import MessageProtocol as ClientProtocol
        client_proto = ClientProtocol()
        
        # Check for magic bytes constant
        if hasattr(ClientProtocol, 'MAGIC_BYTES'):
            print(f"   ✅ Has MAGIC_BYTES: 0x{ClientProtocol.MAGIC_BYTES:08X}")
        else:
            print("   ❌ Missing MAGIC_BYTES constant")
            
        # Check max vector size
        if hasattr(ClientProtocol, 'MAX_REASONABLE_VECTOR_LENGTH'):
            print(f"   ✅ MAX_REASONABLE_VECTOR_LENGTH: {ClientProtocol.MAX_REASONABLE_VECTOR_LENGTH:,}")
        else:
            print("   ❌ Missing MAX_REASONABLE_VECTOR_LENGTH")
            
        # Try encoding a test message
        test_vec = [1.0, 2.0, 3.0]
        encoded = client_proto.encode_handshake(test_vec)
        
        # Check if it starts with magic bytes
        if len(encoded) >= 4:
            magic = struct.unpack('!I', encoded[:4])[0]
            if magic == 0xDEADBEEF:
                print(f"   ✅ Encodes with correct magic bytes")
            else:
                print(f"   ❌ Wrong magic bytes in encoding: 0x{magic:08X}")
        else:
            print("   ❌ Encoded message too short")
            
    except ImportError as e:
        print(f"   ❌ Could not import client protocol: {e}")
    except Exception as e:
        print(f"   ❌ Error testing client protocol: {e}")
    
    # Check if we can simulate a handshake
    print("\n3. Protocol Compatibility Test:")
    try:
        from server.src.communication.protocol import MessageProtocol as ServerProtocol
        sys.path.insert(0, 'client_picarx/src/brainstem')
        from brain_client import MessageProtocol as ClientProtocol
        
        # Client sends handshake
        client_proto = ClientProtocol()
        handshake_data = [1.0, 307212.0, 3.0, 640.0, 480.0]
        client_msg = client_proto.encode_handshake(handshake_data)
        print(f"   Client message size: {len(client_msg)} bytes")
        
        # Server receives handshake
        server_proto = ServerProtocol()
        msg_type, received_data = server_proto.decode_message(client_msg)
        
        if msg_type == 2:  # MSG_HANDSHAKE
            print(f"   ✅ Server correctly decoded handshake")
            print(f"   ✅ Received {len(received_data)} values")
            if len(received_data) >= 2 and received_data[1] == 307212.0:
                print(f"   ✅ Vision data size preserved: {int(received_data[1]):,}")
        else:
            print(f"   ❌ Wrong message type: {msg_type}")
            
    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("If all checks pass (✅), the protocols are synchronized.")
    print("If any fail (❌), you need to:")
    print("  1. Ensure both files have the same MAGIC_BYTES value")
    print("  2. Deploy the updated code to the robot")
    print("  3. Restart both brain server and robot client")

if __name__ == "__main__":
    check_protocol_versions()