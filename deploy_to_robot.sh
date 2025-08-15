#!/bin/bash

# Deploy the updated protocol to the robot
# This ensures the robot uses magic bytes and correct validation limits

if [ -z "$1" ]; then
    echo "Usage: $0 <robot-ip>"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

ROBOT_HOST=$1

echo "========================================="
echo "Deploying Protocol Updates to Robot"
echo "========================================="
echo ""
echo "Target robot: $ROBOT_HOST"
echo ""
echo "This will deploy:"
echo "  • Updated brain_client.py with magic bytes (0xDEADBEEF)"
echo "  • MAX_REASONABLE_VECTOR_LENGTH = 10,000,000"
echo "  • Proper message framing and validation"
echo ""

# First, let's check if we can connect to the robot
echo "Testing connection to robot..."
if ! ping -c 1 -W 2 $ROBOT_HOST > /dev/null 2>&1; then
    echo "❌ Cannot reach robot at $ROBOT_HOST"
    echo "   Please check the IP address and network connection"
    exit 1
fi
echo "✅ Robot is reachable"

# Deploy the updated brain_client.py
echo ""
echo "Deploying updated brain_client.py..."
scp client_picarx/src/brainstem/brain_client.py pi@$ROBOT_HOST:~/em-brain/client_picarx/src/brainstem/brain_client.py

if [ $? -eq 0 ]; then
    echo "✅ File deployed successfully"
else
    echo "❌ Failed to deploy file. Check SSH access to robot."
    exit 1
fi

# Also deploy the server protocol file as a backup (some imports might use it)
echo "Deploying protocol.py as backup..."
ssh pi@$ROBOT_HOST "mkdir -p ~/em-brain/server/src/communication" 2>/dev/null
scp server/src/communication/protocol.py pi@$ROBOT_HOST:~/em-brain/server/src/communication/protocol.py 2>/dev/null

# Now restart the robot client
echo ""
echo "Restarting robot client..."
ssh pi@$ROBOT_HOST << 'ENDSSH'
# Kill any existing robot process
echo "Stopping existing robot process..."
pkill -f "python.*picarx_robot.py" 2>/dev/null
sleep 2

# Check if process stopped
if pgrep -f "python.*picarx_robot.py" > /dev/null; then
    echo "⚠️  Robot process still running, forcing kill..."
    pkill -9 -f "python.*picarx_robot.py" 2>/dev/null
    sleep 1
fi

echo "Robot process stopped"
ENDSSH

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "✅ Updated protocol deployed to robot"
echo ""
echo "Now start the robot manually with:"
echo "  ssh pi@$ROBOT_HOST"
echo "  cd ~/em-brain/client_picarx"
echo "  python3 picarx_robot.py --brain-host <BRAIN-SERVER-IP>"
echo ""
echo "Or run it in background:"
echo "  ssh pi@$ROBOT_HOST 'cd ~/em-brain/client_picarx && nohup python3 picarx_robot.py --brain-host <BRAIN-IP> > robot.log 2>&1 &'"
echo ""
echo "The robot will now use:"
echo "  • Magic bytes (0xDEADBEEF) for message validation"
echo "  • Support for 307,212 sensor values (640x480 vision)"
echo "  • Proper TCP stream synchronization"