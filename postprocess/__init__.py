"""OCT post-processing pipeline: registration, deconvolution, and QC.

Public API
----------
postprocess_stacks : High-level function to register and deconvolve stacks.
run_self_test      : Synthetic validation of registration + deconvolution.
"""
from .pipeline import postprocess_stacks
from .reporting import run_self_test

__all__ = ["postprocess_stacks", "run_self_test"]
