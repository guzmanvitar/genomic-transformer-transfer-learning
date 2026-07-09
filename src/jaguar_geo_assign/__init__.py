"""Top-level package for the jaguar geographic-assignment transfer-learning system.

This package wires together a reproducible feline genomics pipeline (corpus
construction, tokenization, and diagnostics) supporting downstream jaguar
geographic-assignment research via DNABERT-2 transfer learning.  Subpackages
correspond to pipeline stages: ``data`` (contracts and acquisition),
``pretrain`` (corpus preparation and foundation training), and ``fine_tune``
(genotype MLP with VES-guided locus gates).
"""

__version__ = "0.1.0"
