#!/bin/bash
# Minimal installation script for PiCar-X Brain Client
# For use on Raspberry Pi with PiCar-X robot

set -e

echo "🤖 PiCar-X Brain Client - Minimal Install"
echo "=========================================="
echo ""

# Check if on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: Not on Raspberry Pi!"
    read -p "Continue? (y/N): " yn
    [[ ! $yn =~ ^[Yy]$ ]] && exit 1
fi

# 1. Install system dependencies
echo "📦 Installing system packages..."
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev

# 2. Install SunFounder libraries (if not present)
if ! python3 -c "import robot_hat" 2>/dev/null; then
    echo "🔧 Installing robot-hat..."
    cd /tmp
    git clone https://github.com/sunfounder/robot-hat.git
    cd robot-hat
    sudo python3 setup.py install
    cd ..
    rm -rf robot-hat
fi

if ! python3 -c "import picarx" 2>/dev/null; then
    echo "🚗 Installing picar-x..."
    cd /tmp
    git clone -b v2.0 https://github.com/sunfounder/picar-x.git
    cd picar-x
    sudo python3 setup.py install
    cd ..
    rm -rf picar-x
fi

# 3. Setup Python environment
echo "🐍 Setting up Python environment..."
cd ~/client_picarx
python3 -m venv venv
source venv/bin/activate

# 4. Install Python packages
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install numpy psutil

# 5. Create default config if not exists
if [ ! -f config/client_settings.json ]; then
    echo "⚙️  Creating default configuration..."
    cat > config/client_settings.json << EOF
{
    "brain_host": "localhost",
    "brain_port": 9999,
    "robot_profile": "default",
    "safe_mode": true
}
EOF
fi

# 6. Create start script
echo "📝 Creating start script..."
cat > start_robot.sh << 'EOF'
#!/bin/bash
cd $(dirname $0)
source venv/bin/activate
python picarx_robot.py $@
EOF
chmod +x start_robot.sh

echo ""
echo "✅ Installation complete!"
echo ""
echo "To configure brain server:"
echo "  nano config/client_settings.json"
echo ""
echo "To run:"
echo "  ./start_robot.sh --brain-host <IP>"
echo ""
echo "For auto-start on boot, see DEPLOYMENT_GUIDE.md"