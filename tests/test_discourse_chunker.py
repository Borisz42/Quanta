"""Unit tests for DiscourseChunker and DiscourseChunk (Phase 1)."""

import pytest
from parser.chunker import DiscourseChunk, DiscourseChunker, SentenceSpan


ELEANOR_VANCE_NARRATIVE = (
    "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
    "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which "
    "strongly suggested an unobserved phase transition. Although her supervisor initially doubted the validity "
    "of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation "
    "within the same vessel. The resulting polymer retained its structural integrity throughout the afternoon, "
    "prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited."
)

MULTI_PARAGRAPH_NARRATIVE = """The laboratory hummed with the steady pulse of liquid nitrogen pumps. Dr. Eleanor Vance adjusted the focal array of the spectrometer, peering through the frosted viewport. Sunlight was only beginning to crest the horizon outside the facility.

She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which strongly suggested an unobserved phase transition. Her notes quickly filled several pages of her digital logbook. Every metric deviated significantly from standard polymer kinetics.

Although her supervisor initially doubted the validity of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation within the same vessel. The data was unmistakable and consistent.

The resulting polymer retained its structural integrity throughout the afternoon, prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited. A full inquiry was scheduled for the following morning."""


BOOK_TEXT = """# Chapter 1: The Synthesis

Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. She immediately noted that this specimen exhibited anomalous crystalline lattice expansion.

Her supervisor arrived shortly before noon to inspect the containment unit. Together they reviewed the spectral readings from the automated sensor suite.

***

In the second observation bay, Dr. Marcus Wright continued his calibration runs. He observed no such anomalies in his control samples, further confirming Eleanor's findings.

# Chapter 2: The Audit

The formal audit commenced at nine in the morning with three external evaluators present. Eleanor presented her raw sensor logs and crystal diffraction maps.

The review committee spent hours cross-referencing the temperature gradients. By dusk, all members signed the validation certificate without dissent."""


DIALOGUE_TEXT = """Eleanor turned to her colleague with a look of quiet disbelief.
"Did you observe the lattice parameters on sensor four?" she asked.
Marcus leaned closer to the monitor. "I saw them, but that should be thermodynamically impossible at eighty Kelvin."
"Yet here it is," Eleanor whispered. "The bond lengths contracted instead of expanding."
"Then we must verify the cryogenic seals before making any announcement," Marcus warned.
"Agreed. Let us run a second verification cycle right away."
"""


@pytest.fixture
def spacy_chunker():
    return DiscourseChunker(min_words=50, max_words=150, use_spacy=True)


@pytest.fixture
def rule_based_chunker():
    return DiscourseChunker(min_words=50, max_words=150, use_spacy=False)


def test_abbreviation_preservation_eleanor(spacy_chunker, rule_based_chunker):
    """Assert that 'Dr. Eleanor Vance' is not split mid-sentence at 'Dr.'."""
    for chunker in [spacy_chunker, rule_based_chunker]:
        sents = chunker.split_into_sentences(ELEANOR_VANCE_NARRATIVE)
        assert len(sents) == 4
        # First sentence starts with Dr. Eleanor Vance and ends with at dawn.
        first_sent = sents[0][0]
        assert first_sent.startswith("Dr. Eleanor Vance")
        assert first_sent.endswith("at dawn.")


def test_multi_paragraph_chunking(spacy_chunker):
    """Test chunking of multi-paragraph narrative into bounded discourse episodes."""
    chunks = spacy_chunker.chunk_document(MULTI_PARAGRAPH_NARRATIVE)
    assert len(chunks) >= 1

    for chunk in chunks:
        assert isinstance(chunk, DiscourseChunk)
        assert len(chunk.sentence_spans) > 0
        assert chunk.word_count > 0
        assert chunk.token_count_estimate > chunk.word_count
        assert len(chunk.paragraph_indices) > 0

        # Verify all sentences in chunk are contiguous and preserve boundaries
        for span in chunk.sentence_spans:
            assert isinstance(span, SentenceSpan)
            # Verify slice of chunk text matches sentence text
            assert chunk.text[span.start_char:span.end_char] == span.text
            # Verify slice of source document matches sentence text
            assert MULTI_PARAGRAPH_NARRATIVE[span.global_start_char:span.global_end_char] == span.text


def test_dialogue_speaker_turns(spacy_chunker):
    """Test that dialogue turns with quotes and conversational tags are preserved."""
    chunks = spacy_chunker.chunk_document(DIALOGUE_TEXT)
    assert len(chunks) >= 1

    for chunk in chunks:
        for span in chunk.sentence_spans:
            # Check provenance matches
            assert DIALOGUE_TEXT[span.global_start_char:span.global_end_char] == span.text
            assert chunk.text[span.start_char:span.end_char] == span.text


def test_chapter_and_scene_delimiters(spacy_chunker):
    """Test chapter heading and scene break detection and boundary enforcement."""
    chunker = DiscourseChunker(min_words=20, max_words=100, use_spacy=True)
    chunks = chunker.chunk_document(BOOK_TEXT)

    assert len(chunks) >= 2

    # Check chapter identifiers
    chapter_ids = [c.chapter_id for c in chunks if c.chapter_id]
    assert any("chapter_1" in cid for cid in chapter_ids)
    assert any("chapter_2" in cid for cid in chapter_ids)

    # Chunks from Chapter 1 should not carry Chapter 2 sentences
    ch1_chunks = [c for c in chunks if c.chapter_id and "chapter_1" in c.chapter_id]
    ch2_chunks = [c for c in chunks if c.chapter_id and "chapter_2" in c.chapter_id]

    assert len(ch1_chunks) >= 1
    assert len(ch2_chunks) >= 1

    for c in ch1_chunks:
        assert "Audit" not in c.text
        assert "Marcus Wright" in c.text or "Eleanor Vance" in c.text

    for c in ch2_chunks:
        assert "The formal audit" in c.text or "certificate without dissent" in c.text


def test_streaming_vs_batch_consistency(spacy_chunker):
    """Verify that stream_chunks yields identical chunks to chunk_document."""
    batch_chunks = spacy_chunker.chunk_document(BOOK_TEXT)
    stream_chunks = list(spacy_chunker.stream_chunks(BOOK_TEXT))

    assert len(batch_chunks) == len(stream_chunks)
    for b, s in zip(batch_chunks, stream_chunks):
        assert b.chunk_id == s.chunk_id
        assert b.text == s.text
        assert b.chapter_id == s.chapter_id
        assert b.global_offset == s.global_offset
        assert b.global_end_offset == s.global_end_offset
        assert len(b.sentence_spans) == len(s.sentence_spans)


def test_oversized_single_sentence(spacy_chunker):
    """Verify that an oversized sentence exceeding max_words is not split mid-sentence."""
    chunker = DiscourseChunker(min_words=10, max_words=20, use_spacy=True)
    long_sentence = (
        "This is an unusually prolonged and continuous compound sentence describing the intricate details of "
        "the chemical reaction, specifically how the catalyst interacted with the polymer matrix under extreme "
        "cryogenic temperatures, resulting in irreversible conformational transformations that surprised all the researchers."
    )
    assert len(long_sentence.split()) > 20

    chunks = chunker.chunk_document(long_sentence)
    assert len(chunks) == 1
    assert chunks[0].text == long_sentence
    assert len(chunks[0].sentence_spans) == 1
    assert chunks[0].sentence_spans[0].text == long_sentence


def test_custom_window_boundaries():
    """Test configurable min_words and max_words constraints."""
    with pytest.raises(ValueError):
        DiscourseChunker(min_words=300, max_words=100)

    chunker = DiscourseChunker(min_words=30, max_words=80, use_spacy=False)
    chunks = chunker.chunk_document(MULTI_PARAGRAPH_NARRATIVE)
    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk should contain full sentences
        assert len(chunk.sentence_spans) >= 1
        for span in chunk.sentence_spans:
            assert span.text in chunk.text
