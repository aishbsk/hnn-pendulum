"""
This file is automatically loaded by pytest.
It sets up the Python path to include the src directory.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
