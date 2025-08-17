#!/usr/bin/env python3
"""
Test that mimics the EXACT handshake pattern to isolate the issue.
"""

import socket
import struct
import sys
import time

def handshake_server(port=9999):
    """Server that mimics brain server handshake exactly."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    
    print(f"Listening on port {port}")
    
    while True:
        client, addr = server.accept()
        print(f"Client connected from {addr}")
        
        try:
            # 1. Receive handshake (exactly like brain server)
            print("Waiting for handshake header...")
            header = client.recv(13)
            print(f"Received header: {len(header)} bytes")
            
            if len(header) == 13:
                magic, msg_len, msg_type, vec_len = struct.unpack('!IIBI', header)
                print(f"Header: magic=0x{magic:08X}, len={msg_len}, type={msg_type}, vec={vec_len}")
                
                # Receive vector data
                data_size = vec_len * 4
                data = client.recv(data_size)
                print(f"Received data: {len(data)} bytes")
                
                # 2. Send response (exactly like brain server)
                response_capabilities = [1.0, 12.0, 6.0, 1.0, 15.0]
                vector_data = struct.pack('5f', *response_capabilities)
                message_length = 1 + 4 + len(vector_data)
                response_header = struct.pack('!IIBI', 
                    0x524F424F,  # Magic
                    message_length,
                    2,  # MSG_HANDSHAKE
                    5  # Vector length
                )
                response_msg = response_header + vector_data
                
                print(f"Sending response: {len(response_msg)} bytes")
                print(f"Response hex: {response_msg.hex()}")
                
                # Try different send methods
                print("\nMethod 1: sendall()")
                client.sendall(response_msg)
                print("  Sent successfully")
                
                time.sleep(0.1)
                
                print("\nMethod 2: send() with TCP_NODELAY")
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                test_msg = b"TEST123"
                sent = client.send(test_msg)
                print(f"  Sent {sent} bytes")
                
                print("\nKeeping connection open for 5 seconds...")
                time.sleep(5)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            client.close()
            print("Connection closed\n")

def handshake_client(host, port=9999):
    """Client that mimics brain client handshake exactly."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    
    print(f"Connecting to {host}:{port}")
    sock.connect((host, port))
    print("Connected!")
    
    try:
        # 1. Send handshake (exactly like brain client)
        capabilities = [1.0, 12.0, 6.0, 1.0, 3.0, 640.0, 480.0]
        vector_data = struct.pack('7f', *capabilities)
        message_length = 1 + 4 + len(vector_data)
        header = struct.pack('!IIBI',
            0x524F424F,  # Magic
            message_length,
            2,  # MSG_HANDSHAKE
            7  # Vector length
        )
        handshake = header + vector_data
        
        print(f"Sending handshake: {len(handshake)} bytes")
        sock.sendall(handshake)
        print("Handshake sent")
        
        # 2. Receive response (exactly like brain client)
        print("\nWaiting for response header (13 bytes)...")
        sock.settimeout(5.0)
        
        # Try to receive header
        header_data = b''
        while len(header_data) < 13:
            chunk = sock.recv(13 - len(header_data))
            if not chunk:
                print(f"Connection closed after {len(header_data)} bytes")
                break
            header_data += chunk
            print(f"Received chunk: {len(chunk)} bytes, total: {len(header_data)}/13")
        
        if len(header_data) == 13:
            magic, msg_len, msg_type, vec_len = struct.unpack('!IIBI', header_data)
            print(f"Header received: magic=0x{magic:08X}, len={msg_len}, type={msg_type}, vec={vec_len}")
            
            # Receive vector data
            vec_size = vec_len * 4
            vec_data = b''
            while len(vec_data) < vec_size:
                chunk = sock.recv(vec_size - len(vec_data))
                if not chunk:
                    break
                vec_data += chunk
                print(f"Received vector chunk: {len(chunk)} bytes, total: {len(vec_data)}/{vec_size}")
            
            if len(vec_data) == vec_size:
                values = struct.unpack(f'{vec_len}f', vec_data)
                print(f"Response values: {values}")
            else:
                print(f"Incomplete vector data: {len(vec_data)}/{vec_size}")
        else:
            print(f"Incomplete header: {len(header_data)}/13 bytes")
            print(f"Received: {header_data.hex()}")
        
        # Try to receive any additional data
        print("\nChecking for additional data...")
        sock.settimeout(1.0)
        try:
            extra = sock.recv(100)
            if extra:
                print(f"Extra data: {extra}")
        except socket.timeout:
            print("No additional data")
        
    except socket.timeout:
        print("TIMEOUT - no response received!")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()
        print("\nDone")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Server: python test_exact_handshake.py server [port]")
        print("  Client: python test_exact_handshake.py client <host> [port]")
        sys.exit(1)
    
    if sys.argv[1] == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
        handshake_server(port)
    elif sys.argv[1] == "client":
        host = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
        handshake_client(host, port)