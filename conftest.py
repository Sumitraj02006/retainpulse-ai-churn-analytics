"""
conftest.py — shared pytest configuration.
Adds the repo root to sys.path so src imports resolve without installation.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
