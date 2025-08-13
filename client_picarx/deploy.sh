#!/bin/bash
#
# Simple deployment script for PiCar-X robot
# Copies only the necessary files to Raspberry Pi
#

# Configuration
PI_HOST="${PI_HOST:-pi@192.168.1.231}"  # Set PI_HOST env var or edit this
PI_DIR="/home/pi/picarx_robot"

echo "🚀 Simple PiCar-X Deployment"
echo "=================================================="
echo "Target: $PI_HOST:$PI_DIR"
echo ""

# Check if we can connect
echo "Testing connection..."
if ! ssh -q $PI_HOST exit; then
    echo "❌ Cannot connect to $PI_HOST"
    echo "   Set PI_HOST environment variable:"
    echo "   export PI_HOST=pi@192.168.1.xxx"
    exit 1
fi
echo "✅ Connection successful"

# Create directory structure on Pi
echo ""
echo "Creating directory structure..."
ssh $PI_HOST << 'EOF'
    mkdir -p ~/picarx_robot/src/brainstem
    mkdir -p ~/picarx_robot/src/hardware
    mkdir -p ~/picarx_robot/config
EOF

# Copy only necessary files
echo ""
echo "Copying files..."

# Main controller
scp picarx_robot.py $PI_HOST:$PI_DIR/

# Brainstem
scp src/brainstem/brainstem.py $PI_HOST:$PI_DIR/src/brainstem/
scp src/brainstem/brain_client.py $PI_HOST:$PI_DIR/src/brainstem/

# Hardware
scp src/hardware/bare_metal_hal.py $PI_HOST:$PI_DIR/src/hardware/
scp src/hardware/picarx_hardware_limits.py $PI_HOST:$PI_DIR/src/hardware/

# Documentation
scp README.md $PI_HOST:$PI_DIR/
scp INSTALLATION.md $PI_HOST:$PI_DIR/
scp requirements.txt $PI_HOST:$PI_DIR/

# Init files
touch __init__.py
scp __init__.py $PI_HOST:$PI_DIR/src/
scp __init__.py $PI_HOST:$PI_DIR/src/brainstem/
scp __init__.py $PI_HOST:$PI_DIR/src/hardware/
rm __init__.py

# Configuration
echo "{
    \"brain_host\": \"192.168.1.100\",
    \"brain_port\": 9999,
    \"safety\": {
        \"collision_distance_cm\": 10,
        \"battery_critical_v\": 6.0
    }
}" > /tmp/robot_config.json
scp /tmp/robot_config.json $PI_HOST:$PI_DIR/config/

# Create run script on Pi
echo ""
echo "Creating run scripts..."
ssh $PI_HOST << 'EOF'
cd ~/picarx_robot

# Create run script
cat > run.sh << 'SCRIPT'
#!/bin/bash
# Run the robot

# Get brain host from environment or use default
BRAIN_HOST=${BRAIN_HOST:-192.168.1.100}

echo "Starting PiCar-X Robot"
echo "Brain server: $BRAIN_HOST"

# Need sudo for GPIO/I2C access
sudo python3 picarx_robot.py --brain-host $BRAIN_HOST
SCRIPT

chmod +x run.sh

# Create test script
cat > test.sh << 'SCRIPT'
#!/bin/bash
# Test robot hardware

echo "Testing PiCar-X Hardware"
sudo python3 picarx_robot.py --test
SCRIPT

chmod +x test.sh

# Create calibrate script
cat > calibrate.sh << 'SCRIPT'
#!/bin/bash
# Calibrate servos

echo "Calibrating PiCar-X Servos"
sudo python3 picarx_robot.py --calibrate
SCRIPT

chmod +x calibrate.sh

# Create systemd service
sudo tee /etc/systemd/system/picarx-robot.service > /dev/null << 'SERVICE'
[Unit]
Description=PiCar-X Robot Controller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/picarx_robot
Environment="BRAIN_HOST=192.168.1.100"
ExecStart=/usr/bin/python3 /home/pi/picarx_robot/picarx_robot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

echo "✅ Scripts created"
EOF

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "On the Raspberry Pi, you can now:"
echo "  cd ~/picarx_robot"
echo "  ./test.sh          # Test hardware"
echo "  ./calibrate.sh     # Calibrate servos"
echo "  ./run.sh           # Run robot"
echo ""
echo "Or enable auto-start:"
echo "  sudo systemctl enable picarx-robot"
echo "  sudo systemctl start picarx-robot"
echo ""
echo "To set brain server:"
echo "  export BRAIN_HOST=192.168.1.xxx"