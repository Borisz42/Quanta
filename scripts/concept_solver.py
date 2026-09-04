#!/usr/bin/env python3
"""ConceptNet Universal Dimension Solver & Full-Scale Lexical Grounding Engine for QUANTA.

Implements Ontological Density Scoring (ODS) to mathematically evaluate concept quality
and split the ontology into:
- Tier 1 (Core Anchor Universe ~200k concepts): High ODS concepts used by the Incremental
  Sparse Partition Solver to discover the optimal 128 Band 3 & 128 Band 4 discriminative questions.
- Tier 2 (Full Lexical Grounding Universe ~520k concepts): All valid concepts with >= 2
  assertions post-inheritance, fully grounded in 256-byte packed quaternary vectors in SQLite.

Epistemic 4-Valued Belnap Logic Grounding:
- Direct positive -> 1 (TRUE / YES)
- Direct negative & 1st-order transitive negation -> 2 (FALSE / NO / NEGATED)
- 1st-order transitive positive -> 3 (MAYBE / INHERITED)
- 2nd-order transitive & unasserted -> 0 (IRRELEVANT / INACTIVE)
"""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse
import wordfreq

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DUMP_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
LOCAL_DUMP_FILE = "conceptnet-assertions-5.7.0.csv.gz"

# Scaling & Optimization Parameters
TARGET_BAND_QUESTIONS = 256  # 256 universal questions across Band 3 (slots 384-511) & Band 4 (slots 512-639)
TARGET_TIER1_CONCEPTS = 75000  # Top 75k rich concepts with multi-modal affordances & high usage frequency
MAX_CANDIDATE_QUESTIONS = 120000  # Deep candidate pool of 120,000 relations
MAX_SOLVER_ROUNDS = 256  # 256 primary axes
MIN_CONCEPT_ASSERTIONS = 2  # Retention cutoff for Tier-2 grounding lexicon
MIN_EDGE_WEIGHT = 1.0  # Filter out low-confidence assertions
ISA_INHERITANCE_DEPTH = 1  # 1-hop epistemic property inheritance (depth >= 2 discarded)

ALLOWED_POS = {"n", "v", "a", "r"}  # Noun, Verb, Adjective, Adverb

# Low-value linguistic metadata targets to exclude from candidate question dimensions
EXCLUDED_METADATA_TARGETS = {
    "us", "uk", "slang", "archaic", "historical", "pejorative", "informal", "rare",
    "countable", "uncountable", "obsolete", "dialectal", "colloquial", "transitive",
    "intransitive", "plural", "singular", "dated", "british", "american", "nonstandard",
    "noun", "verb", "adjective", "adverb", "regional", "vulgar", "derogatory", "offensive",
    "euphemistic", "humorous", "poetic", "literary", "jargon", "english", "wiktionary"
}

# Negative relation to canonical positive relation mapping
NEGATIVE_RELATION_DUAL_MAP = {
    "/r/NotCapableOf": "/r/CapableOf",
    "/r/NotHasProperty": "/r/HasProperty",
    "/r/NotDesires": "/r/Desires",
    "/r/Antonym": "/r/SimilarTo",
    "/r/DistinctFrom": "/r/IsA",
}

NEGATIVE_RELATIONS = set(NEGATIVE_RELATION_DUAL_MAP.keys())

def is_clean_concept(lemma: str) -> bool:
    """Filters out OCR noise, multi-word phrases (>2 words), and meta-dictionary lemmas."""
    if len(lemma) < 2 or len(lemma) > 32:
        return False
    if not re.match(r"^[a-z]+([ -][a-z]+)?$", lemma):
        return False
    words = lemma.split()
    if len(words) > 2:
        return False
    meta_words = {
        "spelling", "participle", "plural", "inflection", "misspelling", "abbreviation",
        "initialism", "form", "alternative", "variant", "superseded", "misconstruction"
    }
    if any(w in meta_words for w in words):
        return False
    return True

def is_valuable_predicate(rel: str, target: str) -> bool:
    """Filters candidate questions to focus on cognitive, physical, functional, and domain dimensions."""
    if rel in {"/r/FormOf", "/r/DerivedFrom", "/r/EtymologicallyDerivedFrom", "/r/EtymologicallyRelatedTo"}:
        return False
    if rel == "/r/HasContext" and target in EXCLUDED_METADATA_TARGETS:
        return False
    if not is_clean_concept(target):
        return False
    return True

# Actionable/Affordance relations vs weak/passive relations
ACTIONABLE_RELATIONS = {
    "/r/UsedFor",
    "/r/CapableOf",
    "/r/NotCapableOf",
    "/r/ReceivesAction",
    "/r/HasProperty",
    "/r/NotHasProperty",
    "/r/MadeOf",
    "/r/PartOf",
    "/r/HasA",
    "/r/AtLocation",
    "/r/Causes",
    "/r/HasPrerequisite",
    "/r/HasSubevent",
    "/r/HasFirstSubevent",
    "/r/HasLastSubevent",
    "/r/MotivatedByGoal",
    "/r/CausesDesire",
    "/r/Desires",
    "/r/NotDesires",
    "/r/HostsOrContains",
    "/r/HasPart",
    "/r/MaterialFor",
    "/r/CausedBy",
    "/r/PrerequisiteFor",
    "/r/SubjectTo",
    "/r/IsA",
    "/r/DefinedAs",
    "/r/InstanceOf",
}

# Comprehensive relation templates for canonical question phrasing
RELATION_TEMPLATES = {
    # Core Taxonomic & Ontological
    "/r/IsA": "Is it a type of {}?",
    "/r/InstanceOf": "Is it an instance of {}?",
    "/r/DefinedAs": "Is it defined as {}?",
    "/r/dbpedia/genus": "Does it belong to biological genus {}?",
    # Functional & Physical Affordances
    "/r/UsedFor": "Is it used for {}?",
    "/r/CapableOf": "Is it capable of {}?",
    "/r/ReceivesAction": "Can it be {}?",
    "/r/HasProperty": "Is it typically {}?",
    "/r/MadeOf": "Is it composed of {}?",
    "/r/dbpedia/product": "Is it used to produce {}?",
    # Mereological & Structural
    "/r/PartOf": "Is it a component or part of {}?",
    "/r/HasA": "Does it possess or feature {}?",
    # Spatial & Locative
    "/r/AtLocation": "Is it found in, on, or at {}?",
    "/r/LocatedNear": "Is it typically located near {}?",
    "/r/dbpedia/capital": "Is its capital or headquarters {}?",
    # Causal, Process & Temporal
    "/r/Causes": "Does it cause or lead to {}?",
    "/r/HasPrerequisite": "Does it require {} beforehand?",
    "/r/HasSubevent": "Does this process/action involve {}?",
    "/r/HasFirstSubevent": "Does it begin with {}?",
    "/r/HasLastSubevent": "Does it conclude with {}?",
    "/r/Entails": "Does it strictly entail {}?",
    "/r/MannerOf": "Is it a manner/way of {}?",
    # Teleological & Affective Drives
    "/r/MotivatedByGoal": "Is it done to achieve {}?",
    "/r/CausesDesire": "Does it make one want to {}?",
    "/r/Desires": "Does it desire or seek {}?",
    # Domain, Social & Cultural Context
    "/r/HasContext": "Is it used in the context or domain of {}?",
    "/r/CreatedBy": "Is it created or authored by {}?",
    "/r/dbpedia/genre": "Is it associated with genre {}?",
    "/r/dbpedia/occupation": "Is it associated with occupation {}?",
    "/r/dbpedia/language": "Is it associated with language {}?",
    "/r/dbpedia/field": "Is it associated with field {}?",
    "/r/dbpedia/knownFor": "Is it famous or known for {}?",
    "/r/dbpedia/influencedBy": "Was it influenced by {}?",
    "/r/dbpedia/leader": "Is its leader or head {}?",
    "/r/SymbolOf": "Is it a cultural or symbolic representation of {}?",
    # Lexical & Semantic
    "/r/SimilarTo": "Is it semantically similar to {}?",
    "/r/Attribute": "Is it a qualitative attribute of {}?",
    "/r/RelatedTo": "Is it conceptually related to {}?",
    "/r/DerivedFrom": "Is it etymologically derived from {}?",
    "/r/EtymologicallyRelatedTo": "Is it etymologically related to {}?",
    "/r/EtymologicallyDerivedFrom": "Is it root derived from {}?",
    "/r/FormOf": "Is it an inflected grammatical form of {}?",
    "/r/Synonym": "Is it an exact synonym of {}?",
    # Synthetic Bidirectional Inverses
    "/r/HostsOrContains": "Does it contain, host, or house {}?",
    "/r/HasPart": "Does it consist of or include part {}?",
    "/r/MaterialFor": "Is it used as material to make {}?",
    "/r/CausedBy": "Is it caused or triggered by {}?",
    "/r/PrerequisiteFor": "Is it a prerequisite required for {}?",
    "/r/SubjectTo": "Is it subject to or affected by {}?",
    "/r/ContextFor": "Is it a domain or context for {}?",
}

INVERSE_RELATION_MAP = {
    "/r/AtLocation": "/r/HostsOrContains",
    "/r/PartOf": "/r/HasPart",
    "/r/MadeOf": "/r/MaterialFor",
    "/r/Causes": "/r/CausedBy",
    "/r/HasPrerequisite": "/r/PrerequisiteFor",
    "/r/ReceivesAction": "/r/SubjectTo",
    "/r/HasContext": "/r/ContextFor",
}

SYMMETRIC_RELATIONS = {
    "/r/SimilarTo",
    "/r/Antonym",
    "/r/DistinctFrom",
    "/r/LocatedNear",
    "/r/RelatedTo",
    "/r/EtymologicallyRelatedTo",
    "/r/Synonym",
}


# ==============================================================================
# 1. DOWNLOAD & HIGH-PERFORMANCE INTEGER STREAMING PARSER
# ==============================================================================
def download_conceptnet(url: str, output_path: str):
    """Downloads ConceptNet dump if not already present."""
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[*] Found local dump: {output_path} ({size_mb:.1f} MB)", flush=True)
        return

    print(f"[*] Downloading ConceptNet dump from {url}...", flush=True)
    print("    (Compressed archive is ~500MB, please wait)", flush=True)

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(
                f"\r    -> Downloading: {percent}% [{'#' * (percent // 2)}{' ' * (50 - percent // 2)}]"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print("\n[*] Download complete.", flush=True)


def parse_uri(uri: str):
    """Parses a ConceptNet URI like '/c/en/dog/n' or '/c/en/dog' into ('dog', 'n')."""
    parts = uri.split("/")
    if len(parts) >= 4 and parts[2] == "en":
        lemma = parts[3].replace("_", " ").strip().lower()
        if not lemma or len(lemma) > 60:
            return None, None
        pos = parts[4] if len(parts) > 4 and parts[4] in ALLOWED_POS else "n"
        return lemma, pos
    return None, None


def extract_knowledge_graph_fast(filepath: str):
    """Streams ConceptNet dump with zero-overhead integer ID mapping and separate positive/negative assertion tracking."""
    print("[*] Streaming and parsing assertions across full ConceptNet universe...", flush=True)
    concept_vocab = {}  # str -> int
    predicate_vocab = {}  # (str, str) -> int
    inv_concept_vocab = []
    inv_predicate_vocab = []

    edges_c_pos = []
    edges_p_pos = []
    edges_c_neg = []
    edges_p_neg = []
    isa_children = []
    isa_parents = []

    concept_freq = Counter()
    predicate_freq = Counter()
    concept_rel_counts = defaultdict(Counter)

    def get_concept_id(key: str) -> int:
        cid = concept_vocab.get(key)
        if cid is None:
            cid = len(concept_vocab)
            concept_vocab[key] = cid
            inv_concept_vocab.append(key)
        return cid

    def get_predicate_id(rel: str, target: str) -> int:
        pkey = (rel, target)
        pid = predicate_vocab.get(pkey)
        if pid is None:
            pid = len(predicate_vocab)
            predicate_vocab[pkey] = pid
            inv_predicate_vocab.append(pkey)
        return pid

    t0 = time.time()
    line_count = 0
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line_count = line_idx + 1
            if line_idx % 3000000 == 0 and line_idx > 0:
                elapsed = time.time() - t0
                speed_k = (line_idx / elapsed) / 1000.0
                print(
                    f"    Processed {line_idx:,} / 34,000,000 lines ({line_idx/34000000*100:4.1f}% | {speed_k:.1f}k lines/s)...",
                    flush=True,
                )

            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            rel = parts[1]
            if rel not in RELATION_TEMPLATES and rel not in INVERSE_RELATION_MAP and rel not in NEGATIVE_RELATIONS:
                continue

            start_lemma, start_pos = parse_uri(parts[2])
            end_lemma, end_pos = parse_uri(parts[3])

            if not start_lemma or not end_lemma or start_lemma == end_lemma:
                continue

            if not is_clean_concept(start_lemma) or not is_clean_concept(end_lemma):
                continue

            try:
                data = json.loads(parts[4])
                weight = float(data.get("weight", 1.0))
            except Exception:
                weight = 1.0

            if weight < MIN_EDGE_WEIGHT:
                continue

            c_key = f"{start_lemma} ({start_pos})"
            cid = get_concept_id(c_key)
            
            is_neg = rel in NEGATIVE_RELATIONS
            canon_rel = NEGATIVE_RELATION_DUAL_MAP.get(rel, rel)

            if is_valuable_predicate(canon_rel, end_lemma):
                pid = get_predicate_id(canon_rel, end_lemma)
                if is_neg:
                    edges_c_neg.append(cid)
                    edges_p_neg.append(pid)
                else:
                    edges_c_pos.append(cid)
                    edges_p_pos.append(pid)
                concept_freq[cid] += 1
                predicate_freq[pid] += 1
                concept_rel_counts[cid][rel] += 1

            # IsA taxonomic edge
            if rel == "/r/IsA":
                target_key = f"{end_lemma} ({end_pos})"
                parent_id = get_concept_id(target_key)
                isa_children.append(cid)
                isa_parents.append(parent_id)

            # Bidirectional inverse assertions
            if rel in INVERSE_RELATION_MAP:
                inv_rel = INVERSE_RELATION_MAP[rel]
                if is_valuable_predicate(inv_rel, start_lemma):
                    target_key = f"{end_lemma} ({end_pos})"
                    t_cid = get_concept_id(target_key)
                    inv_pid = get_predicate_id(inv_rel, start_lemma)
                    edges_c_pos.append(t_cid)
                    edges_p_pos.append(inv_pid)
                    concept_freq[t_cid] += 1
                    predicate_freq[inv_pid] += 1
                    concept_rel_counts[t_cid][inv_rel] += 1

            # Symmetric relation assertions
            elif rel in SYMMETRIC_RELATIONS:
                is_neg_sym = rel in NEGATIVE_RELATIONS
                canon_sym_rel = NEGATIVE_RELATION_DUAL_MAP.get(rel, rel)
                if is_valuable_predicate(canon_sym_rel, start_lemma):
                    target_key = f"{end_lemma} ({end_pos})"
                    t_cid = get_concept_id(target_key)
                    sym_pid = get_predicate_id(canon_sym_rel, start_lemma)
                    if is_neg_sym:
                        edges_c_neg.append(t_cid)
                        edges_p_neg.append(sym_pid)
                    else:
                        edges_c_pos.append(t_cid)
                        edges_p_pos.append(sym_pid)
                    concept_freq[t_cid] += 1
                    predicate_freq[sym_pid] += 1
                    concept_rel_counts[t_cid][rel] += 1

    total_edges = len(edges_c_pos) + len(edges_c_neg)
    print(
        f"[*] Initial streaming finished in {time.time() - t0:.1f}s ({line_count:,} lines)."
        f" Loaded {total_edges:,} assertions ({len(edges_c_pos):,} pos, {len(edges_c_neg):,} neg) across {len(concept_vocab):,} concepts.",
        flush=True,
    )
    return (
        inv_concept_vocab,
        inv_predicate_vocab,
        edges_c_pos,
        edges_p_pos,
        edges_c_neg,
        edges_p_neg,
        isa_children,
        isa_parents,
        concept_freq,
        predicate_freq,
        concept_rel_counts,
    )


# ==============================================================================
# 2. ONTOLOGICAL DENSITY SCORING (ODS) & TIER COMPUTATION
# ==============================================================================
def compute_ontological_density_scores(
    inv_concept_vocab: list,
    concept_freq: Counter,
    concept_rel_counts: dict,
    isa_children: list,
    isa_parents: list,
):
    """Computes Ontological Density Score (ODS) for all concepts based on degree, entropy, affordances, and taxonomy."""
    print("[*] Computing Ontological Density Scores (ODS) across all concepts...", flush=True)
    t0 = time.time()

    # Track taxonomic depth and parent count
    child_parent_count = Counter(isa_children)

    ods_scores = {}
    for cid, c_key in enumerate(inv_concept_vocab):
        deg = concept_freq.get(cid, 0)
        if deg == 0:
            continue

        rel_dist = concept_rel_counts.get(cid, {})
        num_distinct_rels = len(rel_dist)

        # 1. Relational Shannon Entropy
        entropy = 0.0
        if num_distinct_rels > 1:
            for count in rel_dist.values():
                p = count / deg
                entropy -= p * math.log2(p)

        # 2. Affordance / Actionability ratio
        actionable_count = sum(cnt for rel, cnt in rel_dist.items() if rel in ACTIONABLE_RELATIONS)
        affordance_ratio = actionable_count / deg

        # 3. Taxonomic centrality
        tax_parents = child_parent_count.get(cid, 0)

        # 4. Real-world Wikipedia / Web corpus usage frequency (Zipf scale 0.0 to 8.0)
        lemma_only = c_key.split("(")[0].strip()
        zipf_val = wordfreq.zipf_frequency(lemma_only, "en")
        freq_multiplier = 1.0 + 2.0 * (zipf_val / 8.0)

        # Composite U-ODS formula: ODS * Frequency Multiplier
        score = (
            math.log2(1.0 + deg)
            * (1.0 + 1.2 * entropy)
            * (1.0 + 1.5 * affordance_ratio)
            + 0.5 * min(3, tax_parents)
        ) * freq_multiplier
        ods_scores[cid] = score

    print(f"[*] Computed Usage-Weighted U-ODS scores for {len(ods_scores):,} concepts in {time.time() - t0:.2f}s.", flush=True)
    return ods_scores


# ==============================================================================
# 3. SPARSE MATRIX ASSEMBLY & EPISTEMIC 4-VALUED INHERITANCE
# ==============================================================================
def build_and_inherit_matrices(
    inv_concept_vocab,
    inv_predicate_vocab,
    edges_c_pos,
    edges_p_pos,
    edges_c_neg,
    edges_p_neg,
    isa_children,
    isa_parents,
    concept_freq,
    predicate_freq,
    ods_scores,
    max_candidate_q=40000,
    min_assertions=2,
    inheritance_depth=1,
    target_tier1=TARGET_TIER1_CONCEPTS,
):
    """Builds Tier-2 sparse matrix, executes 4-valued BLAS inheritance, and extracts Tier-1 Core submatrix."""
    print(f"[*] Building full Tier-2 grounding universe (min assertions >= {min_assertions})...", flush=True)
    t0 = time.time()

    # Tier-2 concepts: All concepts with at least min_assertions
    tier2_cids = [cid for cid, count in concept_freq.most_common() if count >= min_assertions]
    valid_pids = [pid for pid, _ in predicate_freq.most_common(max_candidate_q)]

    old_cid_to_t2 = {cid: i for i, cid in enumerate(tier2_cids)}
    old_pid_to_new = {pid: j for j, pid in enumerate(valid_pids)}

    N2 = len(tier2_cids)
    M = len(valid_pids)
    print(f"    -> Tier-2 Full Grounding Universe size: {N2:,} concepts across {M:,} candidate axes.", flush=True)

    # Filter positive edges to Tier-2
    pos_rows, pos_cols = [], []
    for c, p in zip(edges_c_pos, edges_p_pos):
        nc = old_cid_to_t2.get(c)
        np_id = old_pid_to_new.get(p)
        if nc is not None and np_id is not None:
            pos_rows.append(nc)
            pos_cols.append(np_id)

    # Filter negative edges to Tier-2
    neg_rows, neg_cols = [], []
    for c, p in zip(edges_c_neg, edges_p_neg):
        nc = old_cid_to_t2.get(c)
        np_id = old_pid_to_new.get(p)
        if nc is not None and np_id is not None:
            neg_rows.append(nc)
            neg_cols.append(np_id)

    m_pos_0 = sparse.csr_matrix(
        (np.ones(len(pos_rows), dtype=np.float32), (np.array(pos_rows, dtype=np.int32), np.array(pos_cols, dtype=np.int32))),
        shape=(N2, M),
        dtype=np.float32,
    )
    m_pos_0.data = (m_pos_0.data > 0).astype(np.float32)
    m_pos_0.eliminate_zeros()

    m_neg_0 = sparse.csr_matrix(
        (np.ones(len(neg_rows), dtype=np.float32), (np.array(neg_rows, dtype=np.int32), np.array(neg_cols, dtype=np.int32))),
        shape=(N2, M),
        dtype=np.float32,
    )
    m_neg_0.data = (m_neg_0.data > 0).astype(np.float32)
    m_neg_0.eliminate_zeros()

    # Fast BLAS taxonomic inheritance (T @ mat)
    t_isa = time.time()
    valid_isa_c, valid_isa_p = [], []
    for c, p in zip(isa_children, isa_parents):
        nc = old_cid_to_t2.get(c)
        np_id = old_cid_to_t2.get(p)
        if nc is not None and np_id is not None and nc != np_id:
            valid_isa_c.append(nc)
            valid_isa_p.append(np_id)

    if valid_isa_c and inheritance_depth > 0:
        T_data = np.ones(len(valid_isa_c), dtype=np.float32)
        T = sparse.csr_matrix(
            (T_data, (np.array(valid_isa_c, dtype=np.int32), np.array(valid_isa_p, dtype=np.int32))),
            shape=(N2, N2),
            dtype=np.float32,
        )

        print(f"[*] Propagating 4-valued taxonomic property inheritance via sparse matrix products (depth = {inheritance_depth})...", flush=True)
        # 1st-hop positive: M_pos_1 = T . M_pos_0
        m_pos_1 = T.dot(m_pos_0)
        m_pos_1.data = (m_pos_1.data > 0).astype(np.float32)
        m_pos_1.eliminate_zeros()

        # 1st-hop negative: M_neg_1 = T . M_neg_0
        m_neg_1 = T.dot(m_neg_0)
        m_neg_1.data = (m_neg_1.data > 0).astype(np.float32)
        m_neg_1.eliminate_zeros()

        # Disjoint Indicators and Non-Monotonic Precedence
        # 1. Explicit Negation (d_neg in {0, 1}) -> 2 (FALSE)
        m_false = m_neg_0 + m_neg_1
        m_false.data = (m_false.data > 0).astype(np.uint8)
        m_false.eliminate_zeros()

        # 2. Affirmed Direct Positive (d_pos = 0 \ M_false) -> 1 (TRUE)
        m_true = m_pos_0 - m_pos_0.multiply(m_false)
        m_true.eliminate_zeros()
        m_true.data = (m_true.data > 0).astype(np.uint8)

        # 3. 1st-Hop Inherited Positive (d_pos = 1 \ (M_false | M_true)) -> 3 (MAYBE)
        m_claimed = m_false + m_true
        m_maybe = m_pos_1 - m_pos_1.multiply(m_claimed)
        m_maybe.eliminate_zeros()
        m_maybe.data = (m_maybe.data > 0).astype(np.uint8)

        # Combined Quaternary CSR Matrix: 1*TRUE + 2*FALSE + 3*MAYBE
        mat_csr_t2 = (m_true.astype(np.uint8) + (2 * m_false).astype(np.uint8) + (3 * m_maybe).astype(np.uint8)).tocsr()
        mat_csr_t2.eliminate_zeros()

        print(
            f"[*] Epistemic 4-valued inheritance completed in {time.time() - t_isa:.2f}s:\n"
            f"    -> TRUE  (1): {m_true.nnz:,} assertions\n"
            f"    -> FALSE (2): {m_false.nnz:,} assertions\n"
            f"    -> MAYBE (3): {m_maybe.nnz:,} assertions\n"
            f"    -> Total active non-zeros: {mat_csr_t2.nnz:,}",
            flush=True,
        )
    else:
        m_false = m_neg_0
        m_false.data = (m_false.data > 0).astype(np.uint8)
        m_true = m_pos_0 - m_pos_0.multiply(m_false)
        m_true.eliminate_zeros()
        m_true.data = (m_true.data > 0).astype(np.uint8)
        mat_csr_t2 = (m_true.astype(np.uint8) + (2 * m_false).astype(np.uint8)).tocsr()
        mat_csr_t2.eliminate_zeros()

    # Identify Tier-1 Core Anchor Universe based on ODS Pareto ranking
    print("[*] Filtering Tier-1 Core Anchor Universe by Ontological Density Score (ODS)...", flush=True)
    t2_ods_scores = np.array([ods_scores.get(cid, 0.0) for cid in tier2_cids])
    
    sorted_concept_indices = np.argsort(-t2_ods_scores)
    target_n = min(target_tier1, len(sorted_concept_indices))
    tier1_raw_indices = sorted_concept_indices[:target_n]
    N_raw = len(tier1_raw_indices)
    min_ods = t2_ods_scores[tier1_raw_indices[-1]]
    print(
        f"    -> Selected Top {N_raw:,} Rich Core Concepts (ODS >= {min_ods:.2f}) for Question Optimization.",
        flush=True,
    )

    # Deduplication into Canonical Semantic Archetypes
    print("[*] Deduplicating identical quaternary assertion rows into canonical semantic archetypes...", flush=True)
    t_dedup = time.time()
    mat_csr_t1_raw = mat_csr_t2[tier1_raw_indices, :].tocsr()
    indptr = mat_csr_t1_raw.indptr
    indices = mat_csr_t1_raw.indices
    data = mat_csr_t1_raw.data
    
    profile_map = {}
    archetype_subindices = []
    for i in range(N_raw):
        start, end = indptr[i], indptr[i+1]
        row_bytes = indices[start:end].tobytes() + b"|" + data[start:end].tobytes()
        if row_bytes not in profile_map:
            profile_map[row_bytes] = i
            archetype_subindices.append(i)
            
    archetype_indices = tier1_raw_indices[np.array(archetype_subindices, dtype=np.int32)]
    N_arch = len(archetype_indices)
    print(
        f"    -> Deduplicated {N_raw:,} Tier-1 concepts into {N_arch:,} distinct semantic archetypes ({N_arch/N_raw*100:.1f}%) in {time.time() - t_dedup:.2f}s.",
        flush=True,
    )

    mat_csr_t1 = mat_csr_t2[archetype_indices, :].tocsr()
    mat_csc_t1 = mat_csr_t1.tocsc()

    tier2_concepts = [inv_concept_vocab[cid] for cid in tier2_cids]
    tier1_concepts = [tier2_concepts[i] for i in archetype_indices]
    top_predicates = [inv_predicate_vocab[pid] for pid in valid_pids]
    questions = [
        RELATION_TEMPLATES.get(rel, "Is it related to {}?").format(target)
        for rel, target in top_predicates
    ]

    return (
        tier1_concepts,
        tier2_concepts,
        top_predicates,
        questions,
        mat_csr_t1,
        mat_csc_t1,
        mat_csr_t2,
    )


# ==============================================================================
# 4. FAST INCREMENTAL SPARSE PARTITION SOLVER (TIER-1 CORE, 4-VALUED)
# ==============================================================================
def solve_incremental_questions(
    tier1_concepts: list,
    top_predicates: list,
    questions: list,
    mat_csr_t1: sparse.csr_matrix,
    mat_csc_t1: sparse.csc_matrix,
    target_k: int = None,
    max_rounds: int = 10000,
):
    """Ultra-fast inverted-index partition refinement solver with real-world Zipf corpus weights for 4-valued logic."""
    N, M = mat_csr_t1.shape
    target_desc = f"{target_k} questions (Band 3 & 4)" if target_k is not None else "0 collisions (100% full discrimination)"
    print(f"\n[*] Running Vectorized Inverted-Index Partition Solver across {N:,} Tier-1 Core Concepts (4-Valued Belnap Logic)...", flush=True)
    print(f"    (Target: {target_desc})", flush=True)

    print("[*] Computing real-world Zipf corpus weights across Tier-1 concepts...", flush=True)
    weights = np.array([
        max(0.5, wordfreq.zipf_frequency(c.split("(")[0].strip(), "en"))
        for c in tier1_concepts
    ], dtype=np.float64)

    total_weight = float(np.sum(weights))

    # Sparse indicator matrices for values 1 (TRUE), 2 (FALSE), 3 (MAYBE)
    m1_csr = (mat_csr_t1 == 1).astype(np.float32)
    m2_csr = (mat_csr_t1 == 2).astype(np.float32)
    m3_csr = (mat_csr_t1 == 3).astype(np.float32)

    selected_q_indices = []
    selected_mask = np.zeros(M, dtype=bool)

    # Inverted map: concept_idx -> cluster_id
    concept_to_cluster = np.zeros(N, dtype=np.int32)

    def calc_pair_gain(w_tot, w1, w2, w3):
        w0 = w_tot - w1 - w2 - w3
        return w0 * w1 + w0 * w2 + w0 * w3 + w1 * w2 + w1 * w3 + w2 * w3

    # Initial root cluster weights for all candidate questions
    w1_init = np.asarray(m1_csr.T.dot(weights)).ravel().astype(np.float64)
    w2_init = np.asarray(m2_csr.T.dot(weights)).ravel().astype(np.float64)
    w3_init = np.asarray(m3_csr.T.dot(weights)).ravel().astype(np.float64)
    pair_reductions = calc_pair_gain(total_weight, w1_init, w2_init, w3_init)

    root_concept_indices = np.arange(N, dtype=np.int32)
    active_clusters = {0: root_concept_indices}
    cluster_w_map = {0: (total_weight, w1_init, w2_init, w3_init)}
    next_cid = 1

    def count_collisions():
        return sum(len(c) - 1 for c in active_clusters.values() if len(c) > 1)

    initial_collisions = count_collisions()
    print(f"  [+] Initial total colliding concept pairs: {initial_collisions:,}", flush=True)

    round_num = 1
    t_start = time.time()
    max_limit = target_k if target_k is not None else max_rounds

    while len(selected_q_indices) < max_limit and not np.all(selected_mask):
        current_collisions = count_collisions()
        if current_collisions == 0:
            print(
                f"\n[+] SUCCESS: 100% FULL DISCRIMINATION (0 collisions) achieved with {len(selected_q_indices)} questions!",
                flush=True,
            )
            break

        # Find best question
        masked_reductions = pair_reductions.copy()
        masked_reductions[selected_mask] = -1
        best_q = int(np.argmax(masked_reductions))
        best_gain = masked_reductions[best_q]

        if best_gain <= 0:
            twin_clusters = [c for c in active_clusters.values() if len(c) > 1]
            print(f"\n[-] Exhausted discriminative capacity of {M:,} candidate pool.", flush=True)
            print(f"    Remaining colliding concepts: {current_collisions:,} in {len(twin_clusters):,} identical twin clusters.", flush=True)
            break

        selected_q_indices.append(best_q)
        selected_mask[best_q] = True

        # Fast direct column slice from CSC matrix
        col_start = mat_csc_t1.indptr[best_q]
        col_end = mat_csc_t1.indptr[best_q + 1]
        col_concept_indices = mat_csc_t1.indices[col_start:col_end]
        col_values = mat_csc_t1.data[col_start:col_end]

        if len(col_concept_indices) == 0:
            continue

        concept_val_map = dict(zip(col_concept_indices, col_values))
        affected_cids = np.unique(concept_to_cluster[col_concept_indices])

        for cid in affected_cids:
            c = active_clusters.get(cid)
            if c is None or len(c) <= 1:
                continue

            sub_lists = {0: [], 1: [], 2: [], 3: []}
            for idx in c:
                v = concept_val_map.get(idx, 0)
                sub_lists[v].append(idx)

            non_empty_parts = [v for v, lst in sub_lists.items() if len(lst) > 0]
            if len(non_empty_parts) <= 1:
                continue

            w_old, w1_old, w2_old, w3_old = cluster_w_map.pop(cid)
            del active_clusters[cid]

            # Subtract old contribution
            pair_reductions -= calc_pair_gain(w_old, w1_old, w2_old, w3_old)

            sub_arrays = {v: np.array(lst, dtype=np.int32) for v, lst in sub_lists.items() if len(lst) > 0}
            
            # Hopcroft smaller-part optimization: find largest sub-cluster
            largest_v = max(sub_arrays.keys(), key=lambda v: len(sub_arrays[v]))

            sub_w_info = {}
            for v, c_sub in sub_arrays.items():
                if v == largest_v:
                    continue
                w_v = float(np.sum(weights[c_sub]))
                w1_v = np.asarray(m1_csr[c_sub].T.dot(weights[c_sub])).ravel().astype(np.float64)
                w2_v = np.asarray(m2_csr[c_sub].T.dot(weights[c_sub])).ravel().astype(np.float64)
                w3_v = np.asarray(m3_csr[c_sub].T.dot(weights[c_sub])).ravel().astype(np.float64)
                sub_w_info[v] = (w_v, w1_v, w2_v, w3_v)

            # Compute largest sub-cluster by subtraction
            w_lg = w_old - sum(info[0] for info in sub_w_info.values())
            w1_lg = w1_old - sum(info[1] for info in sub_w_info.values())
            w2_lg = w2_old - sum(info[2] for info in sub_w_info.values())
            w3_lg = w3_old - sum(info[3] for info in sub_w_info.values())
            sub_w_info[largest_v] = (w_lg, w1_lg, w2_lg, w3_lg)

            for v, c_sub in sub_arrays.items():
                w_sub, w1_sub, w2_sub, w3_sub = sub_w_info[v]
                sub_cid = next_cid
                next_cid += 1
                active_clusters[sub_cid] = c_sub
                concept_to_cluster[c_sub] = sub_cid
                if len(c_sub) > 1:
                    cluster_w_map[sub_cid] = (w_sub, w1_sub, w2_sub, w3_sub)
                    pair_reductions += calc_pair_gain(w_sub, w1_sub, w2_sub, w3_sub)

        new_collisions = count_collisions()
        resolved_pct = (1.0 - (new_collisions / initial_collisions)) * 100.0
        unique_cnt = sum(1 for c in active_clusters.values() if len(c) == 1)
        unique_pct = (unique_cnt / N) * 100.0

        if round_num <= 20 or round_num % 25 == 0 or new_collisions == 0 or round_num == max_limit:
            print(
                f"  Q#{round_num:04d} (+{best_gain:12.0f} weight | {unique_pct:6.2f}% unique | {resolved_pct:5.2f}% resolved): "
                f"'{questions[best_q]}' -> Collisions: {new_collisions:,}",
                flush=True,
            )
        round_num += 1

    elapsed = time.time() - t_start
    final_collisions = count_collisions()
    unique_concepts = sum(1 for c in active_clusters.values() if len(c) == 1)
    unique_pct = (unique_concepts / N) * 100.0

    print(
        f"\n========================================================"
        f"\n[*] OPTIMIZATION SUMMARY (TIER-1 CORE CONCEPTS: {N:,})"
        f"\n========================================================"
        f"\n  • Total questions selected: {len(selected_q_indices)}"
        f"\n  • Uniquely distinguished concepts: {unique_concepts:,} / {N:,} ({unique_pct:.2f}%)"
        f"\n  • Remaining colliding concepts: {final_collisions:,}"
        f"\n  • Time elapsed: {elapsed:.2f}s (Average: {elapsed / max(1, len(selected_q_indices)) * 1000:.2f}ms / question)"
        f"\n========================================================",
        flush=True,
    )
    return selected_q_indices


# ==============================================================================
# 5. QUANTA EXPORT FUNCTIONS (FULL TIER-2 LEXICON, 4-VALUED)
# ==============================================================================
def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s]", "", text).strip()
    s = re.sub(r"\s+", "_", s).upper()
    return s[:32]


def pack_quaternary_array(arr: np.ndarray, target_dim: int = 1024) -> bytes:
    full_arr = np.zeros(target_dim, dtype=np.uint8)
    n = min(len(arr), target_dim)
    full_arr[:n] = arr[:n] & 0x03
    num_bytes = target_dim // 4
    buf = bytearray(num_bytes)
    for i in range(num_bytes):
        s0 = int(full_arr[4 * i + 0]) & 0x03
        s1 = int(full_arr[4 * i + 1]) & 0x03
        s2 = int(full_arr[4 * i + 2]) & 0x03
        s3 = int(full_arr[4 * i + 3]) & 0x03
        buf[i] = s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)
    return bytes(buf)


def export_quanta_artifacts(
    tier2_concepts: list,
    top_predicates: list,
    questions: list,
    selected_indices: list,
    mat_csr_t2: sparse.csr_matrix,
    output_dir: str = "data",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[*] Exporting QUANTA artifacts across full Tier-2 Lexicon (4-Valued Grounding)...", flush=True)

    # 1. Export QUANTA Slot Registry JSON
    slot_json_path = out_path / "conceptnet_slots.json"
    print(f"[*] Exporting QUANTA Slot Definitions to {slot_json_path}...", flush=True)
    slot_definitions = []
    for rank, q_idx in enumerate(selected_indices):
        rel, target = top_predicates[q_idx]
        q_text = questions[q_idx]
        if rank < 128:
            slot_idx = 384 + rank
            band_num = 3
            category = "ConceptNet Taxonomy & Domain"
        else:
            slot_idx = 512 + (rank - 128)
            band_num = 4
            category = "ConceptNet Affordance & Function"

        slot_name = f"CN_Q{rank+1:03d}_{slugify(target)}"
        slot_def = {
            "index": slot_idx,
            "name": slot_name,
            "band": band_num,
            "category": category,
            "description": q_text,
            "relation": rel,
            "target": target,
        }
        slot_definitions.append(slot_def)

    with open(slot_json_path, "w", encoding="utf-8") as f:
        json.dump(slot_definitions, f, indent=2)
    print(f"    -> Saved {len(slot_definitions)} canonical Band 3 & 4 slot definitions.", flush=True)

    # 2. Export SQLite Grounding Database (Fast Direct Index Slicing)
    sqlite_db_path = out_path / "conceptnet_offline.db"
    print(f"[*] Compiling SQLite Grounding Database to {sqlite_db_path}...", flush=True)
    if sqlite_db_path.exists():
        sqlite_db_path.unlink()

    conn = sqlite3.connect(str(sqlite_db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE concepts (
            key TEXT PRIMARY KEY,
            lemma TEXT,
            pos TEXT,
            active_slots TEXT,
            packed_bytes_hex TEXT
        )
    """)

    db_entries = []
    slot_names = [slot_definitions[i]["name"] for i in range(len(selected_indices))]
    
    # Extract submatrix for selected questions
    sel_submat = mat_csr_t2[:, selected_indices].tocsr()
    indptr = sel_submat.indptr
    indices = sel_submat.indices
    data = sel_submat.data

    t_db = time.time()
    for c_idx, c_key in enumerate(tier2_concepts):
        match = re.match(r"^(.*)\s+\(([navr])\)$", c_key)
        if match:
            lemma, pos = match.group(1), match.group(2)
        else:
            lemma, pos = c_key, "n"

        # Fast direct slice from CSR arrays (zero object creation)
        start, end = indptr[c_idx], indptr[c_idx + 1]
        row_active = indices[start:end]
        row_values = data[start:end]

        active_slots_dict = {slot_names[i]: int(v) for i, v in zip(row_active, row_values)}

        full_vector = np.zeros(1024, dtype=np.uint8)
        for i, v in zip(row_active, row_values):
            if i < 128:
                full_vector[384 + i] = int(v) & 0x03  # Band 3
            elif i < 256:
                full_vector[512 + (i - 128)] = int(v) & 0x03  # Band 4

        packed_hex = pack_quaternary_array(full_vector, 1024).hex()
        slots_json_str = json.dumps(active_slots_dict)

        db_entries.append((f"cn:{c_key}", lemma, pos, slots_json_str, packed_hex))
        if pos == "n" and f"cn:{lemma}" != f"cn:{c_key}":
            db_entries.append((f"cn:{lemma}", lemma, pos, slots_json_str, packed_hex))

        if len(db_entries) >= 50000:
            cur.executemany("INSERT OR REPLACE INTO concepts VALUES (?, ?, ?, ?, ?)", db_entries)
            db_entries = []

    if db_entries:
        cur.executemany("INSERT OR REPLACE INTO concepts VALUES (?, ?, ?, ?, ?)", db_entries)

    conn.commit()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_lemma ON concepts (lemma)")
    conn.commit()
    conn.close()

    db_size_mb = sqlite_db_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {len(tier2_concepts):,} concepts to '{sqlite_db_path}' ({db_size_mb:.2f} MB in {time.time() - t_db:.1f}s).", flush=True)

    # 3. Export Compressed Codebook CSV
    print("[*] Exporting compressed codebook (concept_codebook.csv.gz)...", flush=True)
    codebook_submat = sel_submat.toarray().astype(np.uint8)
    codebook_df = pd.DataFrame(
        codebook_submat,
        index=tier2_concepts,
        columns=[f"Q{i+1}: {questions[q]}" for i, q in enumerate(selected_indices)],
    )
    codebook_df.to_csv(out_path / "concept_codebook.csv.gz", compression="gzip")
    print(f"    -> Saved 'concept_codebook.csv.gz' ({len(selected_indices)} primary dimensions).", flush=True)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUANTA ConceptNet ODS Universal Dimension Solver & Lexicon Engine")
    parser.add_argument("--depth", type=int, default=ISA_INHERITANCE_DEPTH, help="Taxonomic inheritance depth (default: 1)")
    parser.add_argument("--k", type=int, default=TARGET_BAND_QUESTIONS, help="Target number of Band 3 & 4 questions (default: 256)")
    parser.add_argument("--tier1-concepts", type=int, default=TARGET_TIER1_CONCEPTS, help="Number of Tier 1 concepts (default: 75000)")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATE_QUESTIONS, help="Max candidate questions pool (default: 120000)")
    parser.add_argument("--min-assertions", type=int, default=MIN_CONCEPT_ASSERTIONS, help="Min assertions per concept (default: 2)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for artifacts (default: data)")
    parser.add_argument("--dump-file", type=str, default=LOCAL_DUMP_FILE, help="Path to ConceptNet dump file")
    args = parser.parse_args()

    t_global = time.time()
    print("==================================================================", flush=True)
    print("  QUANTA CONCEPTNET ODS UNIVERSAL DIMENSION SOLVER & LEXICON ENGINE", flush=True)
    print("==================================================================", flush=True)

    # 1. Download ConceptNet
    download_conceptnet(DUMP_URL, args.dump_file)

    # 2. Fast Streaming & Integer ID Parsing
    (
        inv_concept_vocab,
        inv_predicate_vocab,
        edges_c_pos,
        edges_p_pos,
        edges_c_neg,
        edges_p_neg,
        isa_children,
        isa_parents,
        concept_freq,
        predicate_freq,
        concept_rel_counts,
    ) = extract_knowledge_graph_fast(args.dump_file)

    # 3. Ontological Density Scoring (ODS)
    ods_scores = compute_ontological_density_scores(
        inv_concept_vocab,
        concept_freq,
        concept_rel_counts,
        isa_children,
        isa_parents,
    )

    # 4. Sparse Matrix Assembly & Epistemic 4-Valued BLAS Inheritance
    (
        tier1_concepts,
        tier2_concepts,
        top_predicates,
        questions,
        mat_csr_t1,
        mat_csc_t1,
        mat_csr_t2,
    ) = build_and_inherit_matrices(
        inv_concept_vocab,
        inv_predicate_vocab,
        edges_c_pos,
        edges_p_pos,
        edges_c_neg,
        edges_p_neg,
        isa_children,
        isa_parents,
        concept_freq,
        predicate_freq,
        ods_scores,
        max_candidate_q=args.max_candidates,
        min_assertions=args.min_assertions,
        inheritance_depth=args.depth,
        target_tier1=args.tier1_concepts,
    )

    # 5. Solve for Multi-Valued Incremental Discrimination on Tier-1 Universe
    selected_indices = solve_incremental_questions(
        tier1_concepts,
        top_predicates,
        questions,
        mat_csr_t1,
        mat_csc_t1,
        target_k=args.k,
        max_rounds=args.k,
    )

    # 6. Export QUANTA Artifacts across Full Tier-2 Grounding Lexicon
    export_quanta_artifacts(
        tier2_concepts,
        top_predicates,
        questions,
        selected_indices,
        mat_csr_t2,
        output_dir=args.output_dir,
    )

    print(f"\n[+] Full 4-valued ODS pipeline completed successfully in {time.time() - t_global:.1f}s.", flush=True)