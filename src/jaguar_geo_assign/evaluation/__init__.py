"""Evaluation stage — scaffold namespace for later implementation.

Will hold metric computation, geodesic-error aggregation, and per-split
evaluation logic once fine-tuned models produce predictions.  Currently
reserved so that the CLI and config validation can reference the evaluation
stage without import errors.
"""
