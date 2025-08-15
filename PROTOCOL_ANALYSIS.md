# Protocol Analysis - What We Actually Know

## Facts (Evidence-Based)

1. **TCP Buffer Limits**
   - Default send buffer: 16KB
   - Default receive buffer: 128KB  
   - System max: 212KB
   - Message size: 1.2MB
   - **Conclusion**: Message is 75x larger than send buffer

2. **Server Behavior**
   - Handshake succeeds
   - Server receives "Receiving large message: 100%"
   - Then client disconnects
   - Different error values each time (0xE1E0E03E, 0x00000000, etc.)

3. **What Works**
   - Small messages (handshake) work fine
   - The protocol format itself is correct
   - Both sides use magic bytes correctly

## The Real Problem

We have a **fundamental design flaw**: Trying to send 1.2MB atomic messages over TCP with 16KB buffers.

TCP is a STREAM protocol, not a MESSAGE protocol. When you send 1.2MB:
- It gets fragmented into ~75 chunks
- `sendall()` blocks until all chunks are sent
- Server might start reading before client finishes sending
- Any timing issue causes desynchronization

## What We've Been Doing Wrong

1. **Adding complexity instead of fixing the root cause**
   - Magic bytes (good, but not the issue)
   - Backward compatibility (unnecessary)
   - Chunked protocol (band-aid)
   - Buffer size increases (system won't allow)

2. **Not addressing the core issue**
   - Sending 307,200 vision pixels at 20Hz = 24MB/s
   - This is not sustainable over TCP
   - Especially not with small buffers

## The Right Solutions (Pick One)

### Option 1: Reduce Data Size
- Downsample vision to 160x120 (19,200 pixels = 77KB)
- Or compress the data
- Or send differences only

### Option 2: Use UDP for Sensor Data
- UDP can handle large datagrams
- Fire-and-forget for sensors
- TCP only for control messages

### Option 3: Proper Streaming Protocol
- Send data in fixed-size chunks (e.g., 8KB)
- Each chunk has sequence number
- Reassemble on receiver side

### Option 4: Message Framing
- Send 4-byte size header
- Then send data in chunks
- Receiver reads size, then reads exactly that many bytes

## Current State Assessment

**We do NOT have a firm grip on the protocol.**

We've been:
- Guessing at solutions
- Not testing systematically  
- Adding layers of complexity
- Not addressing the fundamental mismatch between message size and buffer size

## Recommendation

**STOP adding band-aids. Pick a proper solution:**

1. **Immediate fix**: Reduce vision to 160x120 (77KB fits in buffers)
2. **Proper fix**: Implement proper message framing with size headers
3. **Best fix**: Stream video separately from control data

The current approach of sending 1.2MB messages over TCP with 16KB buffers is fundamentally broken.