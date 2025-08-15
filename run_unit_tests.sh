#!/bin/bash
# Run unit tests for brain infrastructure and mathematical properties

echo "=========================================="
echo "🧪 Running Brain Unit Tests"
echo "=========================================="
echo ""

# Test core infrastructure
echo "📦 Testing Core Infrastructure..."
python3 -m unittest tests.unit.test_core_infrastructure -v 2>&1 | grep -E "test_|OK|FAIL|ERROR" | head -20
echo ""

# Test mathematical properties  
echo "📐 Testing Mathematical Properties..."
python3 -m unittest tests.unit.test_mathematical_properties -v 2>&1 | grep -E "test_|OK|FAIL|ERROR" | head -20
echo ""

# Quick summary
echo "=========================================="
echo "✅ Unit Test Summary"
echo "=========================================="

python3 -c "
import unittest
import sys

loader = unittest.TestLoader()

# Load test suites
infra_suite = loader.loadTestsFromName('tests.unit.test_core_infrastructure')
math_suite = loader.loadTestsFromName('tests.unit.test_mathematical_properties')

# Run tests
runner = unittest.TextTestRunner(verbosity=0)

print('Infrastructure tests:')
infra_result = runner.run(infra_suite)
print(f'  Ran {infra_result.testsRun} tests, Failures: {len(infra_result.failures)}, Errors: {len(infra_result.errors)}')

print('Mathematical tests:')
math_result = runner.run(math_suite)
print(f'  Ran {math_result.testsRun} tests, Failures: {len(math_result.failures)}, Errors: {len(math_result.errors)}')

# Exit with error if any failures
if infra_result.failures or infra_result.errors or math_result.failures or math_result.errors:
    print('\n❌ Some tests failed!')
    sys.exit(1)
else:
    print('\n✅ All tests passed!')
    sys.exit(0)
" 2>/dev/null

exit $?