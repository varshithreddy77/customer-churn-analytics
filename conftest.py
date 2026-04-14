"""Root conftest — makes `src` importable for all tests."""
import sys
from pathlib import Path

# Add the project root to sys.path so `from src.xxx import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
