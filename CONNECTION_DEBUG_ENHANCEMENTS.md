# Connection Debug Enhancements

This document summarizes the comprehensive logging enhancements added to debug connection issues between the brainstem and brain server.

## Problem

The brainstem was showing "Handshake error: Message receive timeout" but the server was silent about connection attempts, making it difficult to debug where the connection was failing.

## Solution

Added comprehensive logging to both client and server sides to track every step of the connection process.

## Enhanced Components

### 1. Server Side Logging

#### `/server/src/communication/clean_tcp_server.py`
- **ACCEPT**: Logs incoming client connections with IP and port
- **CLIENT_ID**: Logs assignment of unique client IDs
- **SOCKET_CONFIG**: Logs socket timeout and buffer configurations
- **RECV_START/SUCCESS**: Logs message reception attempts and success
- **HANDSHAKE_START/COMPLETE**: Logs handshake initiation and completion
- **SEND_START/SUCCESS/FAILED**: Logs response sending with detailed status
- **TIMEOUT/PROTOCOL_ERROR/DISCONNECT**: Logs various error conditions
- **CLEANUP**: Logs connection cleanup procedures

#### `/server/src/communication/protocol.py`
- **RECV_HEADER/BODY**: Logs message header and body reception
- **INVALID_MAGIC**: Logs magic byte validation failures
- **MSG_TOO_SMALL/LARGE**: Logs message size validation
- **DECODE_SUCCESS**: Logs successful message decoding
- **SEND_PROGRESS**: Logs incremental send progress for large messages
- **RECV_PARTIAL**: Logs partial receive operations (normal for large messages)

#### `/server/src/core/connection_handler.py`
- **HANDSHAKE_RECEIVED**: Logs incoming handshake with capabilities
- **ROBOT_REGISTER/SESSION_CREATE**: Logs robot registration and session creation
- **HANDSHAKE_SUCCESS**: Logs successful handshake completion
- **SENSORY_INPUT**: Logs sensory data processing (sampled to avoid spam)
- **DISCONNECT_START/COMPLETE**: Logs client disconnect procedures

### 2. Client Side Logging

#### `/client_picarx/src/brainstem/brain_client.py`

**BrainClient Connection Logging:**
- **CONNECT_START**: Logs connection initiation with host/port
- **SOCKET_CLEANUP/CREATE/CONFIG**: Logs socket lifecycle management
- **TCP_CONNECT**: Logs TCP connection timing and success
- **CONNECT_TIMEOUT/REFUSED**: Logs specific connection failure types

**Handshake Logging:**
- **HANDSHAKE_SEND**: Logs capability data being sent
- **HANDSHAKE_ENCODE**: Logs message encoding details
- **HANDSHAKE_SENT**: Logs handshake transmission timing
- **HANDSHAKE_RECV**: Logs waiting for and receiving brain response
- **HANDSHAKE_RESPONSE**: Logs brain's negotiated parameters
- **DIMENSION_MATCH/MISMATCH**: Logs parameter negotiation results
- **HANDSHAKE_TIMEOUT**: Logs specific timeout conditions

**MessageProtocol Logging:**
- **RECV_START/HEADER/BODY**: Logs message reception steps
- **INVALID_MAGIC**: Logs stream corruption detection
- **MSG_TOO_SMALL/LARGE**: Logs message validation
- **DECODE_SUCCESS**: Logs successful message parsing

**Sensor Processing Logging:**
- **SENSOR_PADDING/TRUNCATE**: Logs dimension adjustments
- **SENSOR_SEND/SENT**: Logs sensor data transmission
- **MOTOR_RECV/SUCCESS**: Logs motor command reception
- **PROCESS_TIMEOUT**: Logs normal timeout conditions

#### `/client_picarx/src/brainstem/brainstem.py`

**Brainstem Connection Management:**
- **BRAIN_INIT**: Logs brain connection initialization
- **VISION_MODE**: Logs whether vision uses UDP or TCP
- **DIMENSIONS**: Logs calculated sensor dimensions
- **CONNECTION_ATTEMPT/SUCCESS/FAILED**: Logs connection results
- **RECONNECT_ATTEMPT**: Logs reconnection attempts with timing

## Log Message Format

All logs use a consistent format:
```
HH:MM:SS [ComponentName] LEVEL: OPERATION: details
```

Examples:
```
14:32:15 [TCPServer-9999] INFO: ACCEPT: New client connection from 192.168.1.100:54321
14:32:15 [BrainClient] INFO: TCP_CONNECT_SUCCESS: Connected in 0.023s
14:32:15 [ConnectionHandler] INFO: HANDSHAKE_SUCCESS: client_1_abc123 complete, response=[1.0, 24.0, 6.0, 1.0, 3.0]
```

## Debug Levels

- **DEBUG**: Detailed step-by-step operations (socket ops, message details)
- **INFO**: Important state changes (connections, handshakes, disconnects)
- **WARNING**: Non-fatal issues (dimension mismatches, timeouts)
- **ERROR**: Failures that break functionality (connection refused, protocol errors)

## Usage

### Enable Logging in Production

Add to your brainstem startup:
```python
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG for more detail
```

### Test Connection Logging

Run the test script:
```bash
python test_connection_logging.py
```

This will show all connection steps whether the server is running or not.

### Common Debug Scenarios

1. **"Connection Refused"**: Server not running or wrong port
   - Look for: `CONNECT_REFUSED` in client logs
   - Check: Server startup logs for `Server listening on...`

2. **"Connection Timeout"**: Network issues or server overloaded
   - Look for: `CONNECT_TIMEOUT` in client logs
   - Check: Network connectivity, server load

3. **"Handshake Timeout"**: Server accepts connection but doesn't respond
   - Look for: `TCP_CONNECT_SUCCESS` followed by `HANDSHAKE_TIMEOUT`
   - Check: Server handshake processing logs

4. **"Protocol Error"**: Message corruption or version mismatch
   - Look for: `INVALID_MAGIC` or `MSG_TOO_LARGE` 
   - Check: Client/server protocol version compatibility

5. **"Dimension Mismatch"**: Client and server have different expectations
   - Look for: `DIMENSION_MISMATCH` in handshake logs
   - Check: Client sensor configuration vs server expectations

## Benefits

1. **Precise Error Location**: Know exactly where in the connection process failure occurs
2. **Timing Information**: Identify slow network operations or timeouts
3. **Protocol Debugging**: See actual message contents and validation steps
4. **Historical Analysis**: Review logs to identify patterns in connection issues
5. **Remote Debugging**: Debug connection issues without physical access to hardware

The enhanced logging provides complete visibility into the connection process, making it much easier to diagnose and fix communication issues between the brainstem and brain server.