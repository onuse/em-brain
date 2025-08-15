#!/bin/bash

# Deploy protocol fixes to both brain server and robot
# This script fixes the TCP message corruption issue

echo "========================================="
echo "Protocol Fix Deployment Script"
echo "========================================="
echo ""
echo "This will deploy the TCP protocol fixes that prevent message corruption."
echo "The fixes add magic bytes (0xDEADBEEF) and proper message framing."
echo ""

# Check if we have the robot host
if [ -z "$1" ]; then
    echo "Usage: $0 <robot-ip>"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

ROBOT_HOST=$1
BRAIN_HOST=$(hostname -I | awk '{print $1}')

echo "Configuration:"
echo "  Brain Server: $BRAIN_HOST (this machine)"
echo "  Robot Client: $ROBOT_HOST"
echo ""

# Prompt for confirmation
read -p "Deploy fixes? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Step 1: Restarting Brain Server with new protocol..."
echo "----------------------------------------"

# Find and kill existing brain.py process
BRAIN_PID=$(pgrep -f "python.*brain.py")
if [ ! -z "$BRAIN_PID" ]; then
    echo "Stopping existing brain server (PID: $BRAIN_PID)..."
    kill $BRAIN_PID
    sleep 2
fi

# Start brain server in background
echo "Starting brain server with updated protocol..."
cd server
nohup python3 brain.py --safe-mode > brain.log 2>&1 &
NEW_PID=$!
echo "Brain server started (PID: $NEW_PID)"
echo "Log: server/brain.log"
cd ..

echo ""
echo "Step 2: Deploying to Robot..."
echo "----------------------------------------"

# Copy the fixed protocol and brain_client to robot
echo "Copying fixed files to robot..."
scp client_picarx/src/brainstem/brain_client.py pi@$ROBOT_HOST:~/em-brain/client_picarx/src/brainstem/
scp server/src/communication/protocol.py pi@$ROBOT_HOST:~/em-brain/client_picarx/src/communication/

# Also copy the protocol to the right location on robot
ssh pi@$ROBOT_HOST "mkdir -p ~/em-brain/client_picarx/src/communication"
scp server/src/communication/protocol.py pi@$ROBOT_HOST:~/em-brain/client_picarx/src/communication/

echo ""
echo "Step 3: Restarting Robot Client..."
echo "----------------------------------------"

# Restart robot client
ssh pi@$ROBOT_HOST << 'ENDSSH'
# Kill existing robot process
pkill -f "python.*picarx_robot.py"
sleep 1

# Start robot client
cd ~/em-brain/client_picarx
echo "Starting robot client with updated protocol..."
nohup python3 picarx_robot.py --brain-host $BRAIN_HOST > robot.log 2>&1 &
echo "Robot client started"
ENDSSH

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "The protocol fixes have been deployed. The system now:"
echo "  ✅ Uses magic bytes (0xDEADBEEF) for message validation"
echo "  ✅ Properly handles partial TCP reads"
echo "  ✅ Can recover from stream corruption"
echo "  ✅ Validates message sizes (max 50MB)"
echo ""
echo "Monitor the logs:"
echo "  Brain: tail -f server/brain.log"
echo "  Robot: ssh pi@$ROBOT_HOST 'tail -f ~/em-brain/client_picarx/robot.log'"
echo ""
echo "If you see any 'Message too large' errors, they should now show"
echo "proper error messages like 'Invalid magic bytes' instead of huge numbers."