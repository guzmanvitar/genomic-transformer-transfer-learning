"""Canonical field contracts for jaguar sample metadata.

Defines the required metadata columns that every jaguar sample manifest must
contain.  These fields are validated at config-load time
(:func:`~jaguar_geo_assign.config.load_experiment_config`) to guarantee that
downstream stages (splitting, evaluation, reporting) receive a consistent
schema.
"""

# The ordering is significant: config validation compares tuples element-wise
# against the values declared in the experiment TOML.
JAGUAR_METADATA_FIELDS = (
    "sample_id",  # unique per sequencing run
    "individual_id",  # biological individual (split unit)
    "locality_id",  # sampling site identifier
    "biome_population_label",  # population/biome label for classification
    "latitude",  # decimal degrees, WGS-84
    "longitude",  # decimal degrees, WGS-84
)

# Fine-tuning metadata fields - subset required for MTL dataset construction
JAGUAR_FINETUNE_METADATA_FIELDS = (
    "sample_id",
    "individual_id",
    "biome_population_label",
    "latitude",
    "longitude",
)

# Canonical biome/population classes for fine-tuning.
# These are the valid values for biome_population_label in the metadata.
# Used for deterministic label encoding and stratification in k-fold splits.
BIOME_CLASSES = (
    "savanna",
    "rainforest",
    "dry_forest",
    "caatinga",
    "cerrado",
)
