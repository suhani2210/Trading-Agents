#!/bin/bash
# Setup script for AI Trading Agents Platform

echo "================================================"
echo "  AI Trading Agents Platform - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "   Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create __init__.py files
echo ""
echo "📝 Creating package structure..."
touch src/__init__.py
touch src/agents/__init__.py
touch src/data/__init__.py
touch src/orchestration/__init__.py
touch src/backtesting/__init__.py

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📄 Creating .env file from template..."
    cp .env.template .env
    echo "   ⚠️  Please edit .env and add your API keys!"
else
    echo ""
    echo "✅ .env file already exists"
fi

# Create directories for data and logs
echo ""
echo "📁 Creating data directories..."
mkdir -p data
mkdir -p logs
mkdir -p notebooks

echo ""
echo "================================================"
echo "  ✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - OPENAI_API_KEY (required)"
echo "   - NEWS_API_KEY (optional but recommended)"
echo ""
echo "2. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "3. Test the installation:"
echo "   python demo.py"
echo ""
echo "4. Launch the web interface:"
echo "   streamlit run web/app.py"
echo ""
echo "================================================"