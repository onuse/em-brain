#!/bin/bash
# Script to clean up the server directory structure

echo "🧹 Cleaning up server directory structure"
echo "========================================"

# Create organized archive structure
echo "📁 Creating archive directories..."
mkdir -p archive/old_server
mkdir -p archive/old_tests
mkdir -p archive/old_analysis
mkdir -p archive/documentation

# Move old server files
echo "📦 Moving deprecated server files..."
deprecated_server_files=(
    "brain_server.py"
    "debug_action_generation.py"
    "debug_energy_sum_cancellation.py"
    "debug_field_energy_calculation.py"
    "debug_tcp_vs_standalone_energy.py"
    "embodied_learning_test.py"
    "long_term_learning_test.py"
    "profile_brain_performance.py"
    "quick_learning_verification.py"
    "test_brain_learning_abilities.py"
    "test_direct_performance_report.py"
    "test_field_logger_variance_fix.py"
    "test_field_logger_variance.py"
    "test_real_tcp_server_energy.py"
)

for file in "${deprecated_server_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" archive/old_server/
        echo "   ✓ Moved $file"
    fi
done

# Move test files that should be in tests directory
echo "📋 Moving test files to proper location..."
test_files=(
    "test_components.py"
    "test_dynamic_client.py"
    "test_experience_storage.py"
    "test_integrated.py"
    "test_logging_integration.py"
    "test_monitoring_server.py"
    "test_persistence_migration.py"
    "test_simple_brain.py"
)

for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" tests/integration/
        echo "   ✓ Moved $file to tests/integration/"
    fi
done

# Move documentation files
echo "📚 Moving documentation files..."
doc_files=(
    "CONFIGURATION_MIGRATION.md"
    "DEPRECATION_LIST.md"
    "MIGRATION_COMPLETE.md"
)

for file in "${doc_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docs/
        echo "   ✓ Moved $file to docs/"
    fi
done

# Move the migration script itself to tools
if [ -f "migrate_to_dynamic.sh" ]; then
    mv migrate_to_dynamic.sh tools/
    echo "   ✓ Moved migrate_to_dynamic.sh to tools/"
fi

# Move old BrainFactory if it exists
if [ -f "src/brain_factory.py" ]; then
    mv src/brain_factory.py archive/old_server/
    echo "   ✓ Moved src/brain_factory.py"
fi

# Move old communication files
if [ -f "src/communication/tcp_server.py" ]; then
    mv src/communication/tcp_server.py archive/old_server/
    echo "   ✓ Moved src/communication/tcp_server.py"
fi

if [ -f "src/communication/monitoring_server.py" ]; then
    mv src/communication/monitoring_server.py archive/old_server/
    echo "   ✓ Moved src/communication/monitoring_server.py"
fi

# Clean up __pycache__ directories
echo "🗑️  Cleaning up cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Removed __pycache__ directories"

# Create a clean structure summary
echo ""
echo "📊 New directory structure:"
echo "   /"
echo "   ├── dynamic_brain_server.py    # Main server entry point"
echo "   ├── settings.json              # Configuration"
echo "   ├── src/                       # Source code"
echo "   │   ├── core/                  # Core components (new architecture)"
echo "   │   ├── brains/                # Brain implementations"
echo "   │   ├── communication/         # Network layer"
echo "   │   ├── persistence/           # Storage layer"
echo "   │   └── utils/                 # Utilities"
echo "   ├── tests/                     # All tests"
echo "   │   ├── unit/                  # Unit tests"
echo "   │   └── integration/           # Integration tests"
echo "   ├── tools/                     # Tools and utilities"
echo "   │   ├── testing/               # Test runners"
echo "   │   ├── analysis/              # Analysis tools"
echo "   │   └── runners/               # Script runners"
echo "   ├── docs/                      # Documentation"
echo "   ├── logs/                      # Log files"
echo "   ├── robot_memory/              # Persistence data"
echo "   └── archive/                   # Old/deprecated files"
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Review and remove the nested 'server/' directory if empty"
echo "   2. Update any imports in remaining files"
echo "   3. Run: python3 dynamic_brain_server.py"
echo "   4. Run: python3 tools/testing/behavioral_test_dynamic.py"