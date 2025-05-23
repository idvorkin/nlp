#!/bin/bash

# Local GitHub Actions Test Script
# This simulates the core steps of the GitHub Actions workflow

set -e  # Exit on any error

echo "🚀 Starting local GitHub Actions test simulation..."
echo ""

# Step 1: Check Python version (similar to setup-python action)
echo "📋 Step 1: Checking Python version..."
python3 --version
uv --version
echo "✅ Python and uv are available"
echo ""

# Step 2: Install dependencies (same as workflow)
echo "📦 Step 2: Installing dependencies with uv sync..."
uv sync
echo "✅ Dependencies installed"
echo ""

# Step 3: Run tests with pytest (exact same command as workflow)
echo "🧪 Step 3: Running tests with pytest..."
uv run pytest tests/ --verbose --tb=short --junit-xml=pytest.xml
echo "✅ Tests completed"
echo ""

# Step 4: Check test results (simulate what publish-unit-test-result-action does)
echo "📊 Step 4: Analyzing test results..."
if [ -f pytest.xml ]; then
    # Extract test counts from pytest XML (same logic as enhanced workflow)
    TESTS=$(grep -o 'tests="[0-9]*"' pytest.xml | head -1 | sed 's/tests="\([0-9]*\)"/\1/')
    FAILURES=$(grep -o 'failures="[0-9]*"' pytest.xml | head -1 | sed 's/failures="\([0-9]*\)"/\1/')
    ERRORS=$(grep -o 'errors="[0-9]*"' pytest.xml | head -1 | sed 's/errors="\([0-9]*\)"/\1/')
    
    # Default to 0 if not found
    TESTS=${TESTS:-0}
    FAILURES=${FAILURES:-0}
    ERRORS=${ERRORS:-0}
    
    PASSED=$((TESTS - FAILURES - ERRORS))
    
    echo "📈 Test Summary:"
    echo "   Total tests: $TESTS"
    echo "   Passed: $PASSED"
    echo "   Failed: $((FAILURES + ERRORS))"
    
    if [ $((FAILURES + ERRORS)) -eq 0 ]; then
        echo "   Status: ✅ All tests passed!"
        BADGE_STATUS="✅ $PASSED/$TESTS passing"
        BADGE_COLOR="brightgreen"
    else
        echo "   Status: ❌ Some tests failed!"
        BADGE_STATUS="❌ $PASSED/$TESTS passing"
        BADGE_COLOR="red"
    fi
    
    echo "   Badge would show: $BADGE_STATUS"
else
    echo "❌ No pytest.xml file found!"
    exit 1
fi
echo ""

# Step 5: Simulate artifact upload
echo "📤 Step 5: Simulating artifact upload..."
echo "Would upload: pytest.xml ($(wc -c < pytest.xml) bytes)"
echo "✅ Artifacts ready for upload"
echo ""

echo "🎉 Local GitHub Actions simulation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Review pytest.xml for detailed test results"
echo "   2. Push changes to GitHub to trigger the real workflow"
echo "   3. Check the Actions tab in your repository"
echo ""
echo "🔗 Your badge will show: $BADGE_STATUS" 