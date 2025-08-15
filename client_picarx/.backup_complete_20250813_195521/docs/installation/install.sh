#!/bin/bash
"""
PiCar-X Brain Client Installation Script

Complete installation script for deploying the PiCar-X brain client to Raspberry Pi Zero 2 WH.
This installs both SunFounder hardware dependencies and the brain client system.

Prerequisites:
- Raspberry Pi Zero 2 WH with fresh Raspberry Pi OS
- Internet connection configured
- SSH access enabled

Usage:
    git clone https://github.com/onuse/picarx-brain-client.git
    cd picarx-brain-client
    chmod +x docs/installation/install.sh
    ./docs/installation/install.sh
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "🤖 PiCar-X Brain Client Installation"
echo "====================================="
echo -e "${NC}"
echo "This script will install:"
echo "  • SunFounder PiCar-X hardware drivers"
echo "  • Python dependencies for brain client"
echo "  • Auto-start systemd service"
echo "  • Configuration files"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Not running on Raspberry Pi hardware${NC}"
    echo "   This script is designed for Raspberry Pi Zero 2 WH"
    read -p "   Continue anyway? (y/N): " continue_anyway
    if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

# Get current directory (should be project root)
PROJECT_ROOT=$(pwd)
INSTALL_DIR="$PROJECT_ROOT/docs/installation"

echo -e "${BLUE}📍 Installation directory: $PROJECT_ROOT${NC}"
echo ""

# Step 1: SunFounder Hardware Setup
echo -e "${GREEN}🔧 Step 1: SunFounder Hardware Setup${NC}"
echo "Installing PiCar-X hardware drivers and dependencies..."

if [ -f "$INSTALL_DIR/sunfounder_setup.sh" ]; then
    chmod +x "$INSTALL_DIR/sunfounder_setup.sh"
    bash "$INSTALL_DIR/sunfounder_setup.sh"
else
    echo -e "${RED}❌ Error: sunfounder_setup.sh not found in $INSTALL_DIR${NC}"
    exit 1
fi

echo ""

# Step 2: Python Dependencies
echo -e "${GREEN}🐍 Step 2: Python Dependencies${NC}"
echo "Installing Python packages for brain client..."

if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip3 install -r "$PROJECT_ROOT/requirements.txt"
    echo "✅ Python dependencies installed"
else
    echo -e "${YELLOW}⚠️  Warning: requirements.txt not found, installing basic dependencies...${NC}"
    pip3 install numpy requests
fi

echo ""

# Step 3: Configuration
echo -e "${GREEN}⚙️  Step 3: Configuration Setup${NC}"
echo "Setting up configuration files..."

# Create config directory if it doesn't exist
CONFIG_DIR="$PROJECT_ROOT/config"
mkdir -p "$CONFIG_DIR"

# Create default client settings if not exists
if [ ! -f "$CONFIG_DIR/client_settings.json" ]; then
    cat > "$CONFIG_DIR/client_settings.json" << 'EOF'
{
    "brain_server": {
        "host": "192.168.1.100",
        "port": 8080,
        "connection_timeout": 10,
        "retry_interval": 5
    },
    "hardware": {
        "max_speed": 50,
        "max_steering_angle": 30,
        "ultrasonic_safety_distance": 10,
        "camera_enabled": true,
        "vocal_enabled": true
    },
    "logging": {
        "level": "INFO",
        "log_to_file": true,
        "log_directory": "/home/pi/picarx_logs"
    },
    "robot_identity": {
        "robot_id": "picarx_001",
        "robot_type": "picar-x",
        "location": "lab"
    }
}
EOF
    echo "✅ Default configuration created: $CONFIG_DIR/client_settings.json"
    echo -e "${YELLOW}📝 Please edit the brain_server host IP address in client_settings.json${NC}"
else
    echo "✅ Configuration file already exists"
fi

# Create log directory
LOG_DIR="/home/pi/picarx_logs"
mkdir -p "$LOG_DIR"
echo "✅ Log directory created: $LOG_DIR"

echo ""

# Step 4: Systemd Service (Auto-start on boot)
echo -e "${GREEN}🚀 Step 4: Auto-start Service${NC}"
echo "Setting up systemd service for automatic startup..."

# Create systemd service file
sudo tee /etc/systemd/system/picarx-brain.service > /dev/null << EOF
[Unit]
Description=PiCar-X Brain Client
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_ROOT
ExecStart=/usr/bin/python3 $PROJECT_ROOT/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=PYTHONPATH=$PROJECT_ROOT/src

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable picarx-brain.service

echo "✅ Systemd service installed and enabled"
echo ""

# Step 5: Permissions and Final Setup
echo -e "${GREEN}🔐 Step 5: Permissions and Final Setup${NC}"
echo "Setting up file permissions..."

# Make scripts executable
find "$PROJECT_ROOT" -name "*.sh" -exec chmod +x {} \;

# Create main.py if it doesn't exist (entry point)
if [ ! -f "$PROJECT_ROOT/main.py" ]; then
    cat > "$PROJECT_ROOT/main.py" << 'EOF'
#!/usr/bin/env python3
"""
PiCar-X Brain Client Main Entry Point

This script starts the complete brain client system including:
- Hardware initialization
- Brain server connection
- Sensor monitoring
- Motor control
- Vocal expression system
"""

import sys
import os
import time
import signal
import json

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_config():
    """Load configuration from client_settings.json."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'client_settings.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("   Run install.sh to create default configuration")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    # TODO: Add cleanup code here
    sys.exit(0)

def main():
    """Main entry point for PiCar-X brain client."""
    
    print("🤖 PiCar-X Brain Client Starting...")
    print("==================================")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = load_config()
    print(f"✅ Configuration loaded")
    print(f"   Robot ID: {config['robot_identity']['robot_id']}")
    print(f"   Brain Server: {config['brain_server']['host']}:{config['brain_server']['port']}")
    
    # TODO: Initialize hardware, brain client, and start main loop
    print("🔧 Hardware initialization - TODO")
    print("🧠 Brain server connection - TODO") 
    print("🎵 Vocal system initialization - TODO")
    print("📹 Camera system initialization - TODO")
    print("🚗 Motor system initialization - TODO")
    
    print("\n⚠️  This is a placeholder main.py")
    print("   Replace with actual brain client implementation")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Brain client shutting down...")

if __name__ == "__main__":
    main()
EOF
    chmod +x "$PROJECT_ROOT/main.py"
    echo "✅ Created placeholder main.py entry point"
else
    echo "✅ main.py already exists"
fi

echo ""

# Installation Complete
echo -e "${GREEN}"
echo "🎉 PiCar-X Brain Client Installation Complete!"
echo "=============================================="
echo -e "${NC}"
echo "📋 What was installed:"
echo "   ✅ SunFounder PiCar-X hardware drivers"
echo "   ✅ Python dependencies for brain client"
echo "   ✅ Configuration files and logging setup"
echo "   ✅ Auto-start systemd service"
echo "   ✅ File permissions and entry point"
echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo "   1. Edit config/client_settings.json with your brain server IP"
echo "   2. Replace main.py with actual brain client implementation"
echo "   3. Test the installation:"
echo "      sudo systemctl start picarx-brain.service"
echo "      sudo systemctl status picarx-brain.service"
echo "   4. Reboot to test auto-start:"
echo "      sudo reboot"
echo ""
echo -e "${BLUE}🔧 Useful Commands:${NC}"
echo "   sudo systemctl status picarx-brain.service    # Check service status"
echo "   sudo systemctl restart picarx-brain.service   # Restart service"
echo "   sudo journalctl -u picarx-brain.service -f    # View live logs"
echo "   sudo systemctl disable picarx-brain.service   # Disable auto-start"
echo ""
echo -e "${YELLOW}⚠️  Remember to configure the brain server IP in client_settings.json!${NC}"