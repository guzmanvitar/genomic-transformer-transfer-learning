"""Fine-tuning stage — scaffold namespace for later implementation.

Will contain the DNABERT-2 fine-tuning loop, data-loader construction, and
checkpoint management once the pretrained corpus is ready and downstream
jaguar geographic-assignment labels are available.  Currently reserved so
that the CLI and config validation can reference the fine-tune stage without
import errors.
"""
