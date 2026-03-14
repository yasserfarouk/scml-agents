"""Pytest configuration for scml-agents tests."""

import os
import sys

# Fix threading issues with PyTorch/MKL on macOS that cause segfaults
# during orthogonal weight initialization in stable-baselines3
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
