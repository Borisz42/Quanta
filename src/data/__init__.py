"""QUANTA Data: Validation corpus generator, benchmark datasets, and knowledge ingesters."""

from data.corpus_generator import ValidationCorpusGenerator, GeneratedCorpus
from data.wikidata_ingester import (
    SALIENT_PROPERTIES,
    PROPERTY_NAME_TO_PID,
    classify_entity_category,
    stream_wikidata_dump,
    parse_raw_wikidata_entity,
    generate_synthetic_wikidata_slice,
    WikidataEntityMapper,
    WikidataSqliteCompiler,
)

__all__ = [
    "ValidationCorpusGenerator",
    "GeneratedCorpus",
    "SALIENT_PROPERTIES",
    "PROPERTY_NAME_TO_PID",
    "classify_entity_category",
    "stream_wikidata_dump",
    "parse_raw_wikidata_entity",
    "generate_synthetic_wikidata_slice",
    "WikidataEntityMapper",
    "WikidataSqliteCompiler",
]
