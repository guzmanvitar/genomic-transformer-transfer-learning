"""Canonical field contracts for jaguar sample metadata.

Defines the required metadata columns that every jaguar sample manifest must
contain.
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
