#!/usr/bin/env python3
"""
ADAPT Smart Indexing Engine - Quick Start Script

This script provides a quick way to run the ADAPT system.
It can start either the Streamlit frontend or the FastAPI backend.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import fastapi
        import pandas
        import numpy
        import yfinance
        import plotly
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def run_streamlit():
    """Run the Streamlit frontend."""
    print("Starting ADAPT Streamlit Frontend...")
    print("The application will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit server stopped")

def run_fastapi():
    """Run the FastAPI backend."""
    print("Starting ADAPT FastAPI Backend...")
    print("The API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nFastAPI server stopped")

def run_tests():
    """Run the test suite."""
    print("Running ADAPT System Tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_adapt.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing ADAPT dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ADAPT Smart Indexing Engine")
    parser.add_argument(
        "command",
        choices=["frontend", "backend", "test", "install"],
        help="Command to run"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies before running"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ADAPT Smart Indexing Engine")
    print("=" * 50)
    
    # Check dependencies if requested
    if args.check_deps and not check_dependencies():
        sys.exit(1)
    
    # Run the requested command
    if args.command == "frontend":
        run_streamlit()
    elif args.command == "backend":
        run_fastapi()
    elif args.command == "test":
        success = run_tests()
        sys.exit(0 if success else 1)
    elif args.command == "install":
        success = install_dependencies()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 