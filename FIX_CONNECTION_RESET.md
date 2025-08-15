# Fix for Connection Reset & Message Corruption

## The Problem
Server shows: `❌ Unexpected error handling client_2_d4be6879: Message too large: 1695496766`
Client shows: `❌ Communication error: [Errno 104] Connection reset by peer`

## Root Causes

1. **Socket not closed on error**: When the client got an error, it marked `connected = False` but didn't close the socket, leaving corrupted data in the buffer
2. **No fresh socket on reconnect**: Reconnection attempts reused the corrupted socket
3. **Nagle's algorithm**: TCP buffering caused partial messages to be sent

## Fixes Applied

### 1. Client Socket Management (`brain_client.py`)
```python
# On error - CLOSE the socket
except Exception as e:
    print(f"❌ Communication error: {e}")
    self.connected = False
    # IMPORTANT: Close the socket to prevent corruption
    if self.socket:
        try:
            self.socket.close()
        except:
            pass
        self.socket = None  # Clear reference
```

### 2. Fresh Socket on Connect
```python
def connect(self):
    # Close any existing socket first
    if self.socket:
        try:
            self.socket.close()
        except:
            pass
        self.socket = None
    
    # Create fresh socket
    self.socket = socket.socket(...)
```

### 3. Disable Nagle's Algorithm
```python
# For real-time communication
self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
```

### 4. Server Protocol Validation
- Maximum message size for old protocol: 1.5MB (enough for 307K values)
- Proper validation before attempting to read huge amounts

## Deployment

### On Robot:
```bash
cd ~/em-brain
git pull
# Restart robot client
```

### On Server:
```bash
./restart_with_compat.sh
```

## Testing
Use the test script to verify:
```bash
python3 test_direct_connection.py large
```

This should successfully send 307,212 sensor values without corruption.

## What This Fixes
- ✅ No more "Message too large" errors with huge numbers
- ✅ No more "Connection reset by peer" errors
- ✅ Clean reconnection after errors
- ✅ Proper handling of large vision data (1.2MB messages)
- ✅ Real-time communication without buffering delays