"""OCT post-processing pipeline: registration and QC.

Public API
----------
postprocess_stacks : High-level function to register stacks.
run_self_test      : Synthetic validation of registration.
"""
from .pipeline import postprocess_stacks
from .reporting import run_self_test

__all__ = ["postprocess_stacks", "run_self_test"]
