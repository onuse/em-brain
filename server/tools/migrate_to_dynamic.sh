#!/bin/bash
# Migration script to move deprecated files to archive

echo "🚀 Migrating to Dynamic Brain Architecture"
echo "=========================================="

# Create archive directory
echo "📁 Creating archive directory..."
mkdir -p archive/old_architecture/src/communication
mkdir -p archive/old_architecture/tests/integration
mkdir -p archive/old_architecture/tools/testing

# Move main server files
echo "📦 Moving deprecated server files..."
if [ -f "brain_server.py" ]; then
    mv brain_server.py archive/old_architecture/
    echo "   ✓ Moved brain_server.py"
fi

if [ -f "src/brain_factory.py" ]; then
    mv src/brain_factory.py archive/old_architecture/src/
    echo "   ✓ Moved src/brain_factory.py"
fi

# Move communication files
echo "📡 Moving deprecated communication files..."
if [ -f "src/communication/tcp_server.py" ]; then
    mv src/communication/tcp_server.py archive/old_architecture/src/communication/
    echo "   ✓ Moved src/communication/tcp_server.py"
fi

if [ -f "src/communication/monitoring_server.py" ]; then
    mv src/communication/monitoring_server.py archive/old_architecture/src/communication/
    echo "   ✓ Moved src/communication/monitoring_server.py"
fi

# Move test files that use old architecture
echo "🧪 Moving deprecated test files..."
if [ -f "tests/integration/test_brain_server.py" ]; then
    mv tests/integration/test_brain_server.py archive/old_architecture/tests/integration/
    echo "   ✓ Moved tests/integration/test_brain_server.py"
fi

# List of behavioral test files to move
behavioral_tests=(
    "behavioral_test.py"
    "behavioral_test_fast.py"
    "behavioral_test_framework.py"
    "behavioral_test_single_cycle.py"
    "run_standard_behavioral_test.py"
    "run_standard_behavioral_test_quiet.py"
    "run_quick_behavioral_test.py"
)

for test_file in "${behavioral_tests[@]}"; do
    if [ -f "tools/testing/$test_file" ]; then
        mv "tools/testing/$test_file" archive/old_architecture/tools/testing/
        echo "   ✓ Moved tools/testing/$test_file"
    fi
done

echo ""
echo "✅ Migration complete!"
echo ""
echo "📋 Summary:"
echo "   - Main server files moved to archive"
echo "   - Old communication layer moved to archive"
echo "   - Deprecated test files moved to archive"
echo ""
echo "🎯 Next steps:"
echo "   1. Run the new dynamic server: python3 dynamic_brain_server.py"
echo "   2. Run the new behavioral test: python3 tools/testing/behavioral_test_dynamic.py"
echo "   3. Update any remaining analysis tools to use new architecture"
echo ""
echo "⚠️  Note: Many files in tools/analysis/ still use old BrainFactory"
echo "   These need to be updated to use the new dynamic architecture"