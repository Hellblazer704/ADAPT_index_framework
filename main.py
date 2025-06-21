"""
ADAPT Smart Indexing Engine - Main Entry Point
"""

import subprocess
import sys

def main():
    """Start the Streamlit application."""
    print("Starting ADAPT Smart Indexing Engine...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py", 
        "--server.port", "5000", 
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()
