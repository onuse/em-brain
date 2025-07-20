#!/bin/bash
# Field Brain Deployment Script

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../server"
SETTINGS_FILE="$SERVER_DIR/settings.json"
FIELD_SETTINGS="$SERVER_DIR/settings_field_brain.json"
BACKUP_FILE="$SERVER_DIR/settings_backup.json"

echo "🧠 Field Brain Deployment Script"
echo "================================"

# Backup current settings
if [ -f "$SETTINGS_FILE" ]; then
    cp "$SETTINGS_FILE" "$BACKUP_FILE"
    echo "✅ Backed up current settings to settings_backup.json"
fi

# Switch to field brain
if [ -f "$FIELD_SETTINGS" ]; then
    cp "$FIELD_SETTINGS" "$SETTINGS_FILE"
    echo "✅ Switched to field brain configuration"
    echo "📝 Brain type: field"
    echo "🔧 Spatial resolution: 20"
    echo "⏱️  Temporal window: 10.0s"
    echo ""
    echo "🚀 Ready to start server with: python3 brain_server.py"
else
    echo "❌ Field brain settings file not found: $FIELD_SETTINGS"
    exit 1
fi
