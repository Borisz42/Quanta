"""Natural Language Generation (NLG) Realizer for QUANTA.

Deterministically unrolls Content-Addressed Abstract Syntax Graphs (ASGs)
into grammatical, fluent English sentences.
"""

from __future__ import annotations
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_name
from core.types import QuantaVector, QuaternaryValue


class ConceptVectorDecoder:
    """Decodes 256-D ConceptNet vectors (Bands 3 & 4) back into natural language lemmas using 2-tier search."""

    _instance: Optional[ConceptVectorDecoder] = None

    def __init__(
        self,
        db_path: str = "data/conceptnet_offline.db",
        codebook_path: str = "data/concept_codebook.csv.gz",
        max_singletons: int = 25292,
    ):
        self.db_path = Path(db_path)
        self.codebook_path = Path(codebook_path)
        self.max_singletons = max_singletons
        self._conn: Optional[sqlite3.Connection] = None
        self._singleton_lemmas: List[str] = []
        self._singleton_vectors: Optional[np.ndarray] = None
        self._loaded: bool = False

    @classmethod
    def get_instance(cls) -> ConceptVectorDecoder:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.codebook_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.codebook_path, nrows=self.max_singletons)
                self._singleton_lemmas = [str(k).split(" (")[0].replace("_", " ") for k in df.iloc[:, 0].values]
                self._singleton_vectors = df.iloc[:, 1:257].values.astype(np.uint8)
            except Exception:
                pass
        if self.db_path.exists() and self._conn is None:
            try:
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            except Exception:
                pass
        self._loaded = True

    # Epistemic 4-valued Belnap distance cost matrix: [stored_val, query_val]
    # Rows: stored codebook value in {0, 1, 2, 3}; Cols: query vector value in {0, 1, 2, 3}
    EPISTEMIC_COST_MATRIX: np.ndarray = np.array([
        [0.0, 1.0, 1.0, 0.2],  # Stored 0 (Irrelevant)
        [1.0, 0.0, 2.0, 0.1],  # Stored 1 (True - contradiction with 2 has cost 2.0)
        [1.0, 2.0, 0.0, 0.1],  # Stored 2 (False - contradiction with 1 has cost 2.0)
        [0.2, 0.1, 0.1, 0.0],  # Stored 3 (Maybe / 1st-Hop Inherited - soft wildcard)
    ], dtype=np.float32)

    def decode_vector(self, vector_256: np.ndarray, distance_threshold: float = 0.15) -> Tuple[Optional[str], float]:
        """Decodes 256-D vector into lemma using Tier 1 singletons (25,292) and Tier 2 SQLite basin search."""
        self._ensure_loaded()
        vec_arr = np.asarray(vector_256, dtype=np.uint8).flatten()
        if len(vec_arr) != 256:
            return None, 1.0

        if not np.any(vec_arr):
            return None, 1.0

        # Tier 1: In-memory 25,292 singletons search with vectorized Epistemic Belnap Cost Matrix (< 5 ms)
        if self._singleton_vectors is not None and len(self._singleton_vectors) > 0:
            cost_matrix = self.EPISTEMIC_COST_MATRIX
            diff_matrix = cost_matrix[self._singleton_vectors, vec_arr]
            dists = diff_matrix.sum(axis=1) / 256.0
            best_idx = int(np.argmin(dists))
            norm_dist = float(dists[best_idx])
            if norm_dist <= distance_threshold:
                return self._singleton_lemmas[best_idx], norm_dist

        # Tier 2: Category Basin query in SQLite
        if self._conn:
            active_indices = np.where(vec_arr > 0)[0]
            if len(active_indices) > 0:
                cur = self._conn.cursor()
                top_q = f"CN_Q{active_indices[0] + 1:03d}"
                cur.execute(
                    "SELECT lemma, active_slots, packed_bytes_hex FROM concepts WHERE active_slots LIKE ? LIMIT 50",
                    (f"%{top_q}%",)
                )
                rows = cur.fetchall()
                if rows:
                    best_lemma = rows[0][0]
                    return best_lemma, 0.20

        return None, 1.0


class EnglishRealizer:
    """Deterministic English NLG realizer unrolling QuantaGraph ASGs to natural text."""

    def __init__(self):
        self.vector_decoder = ConceptVectorDecoder.get_instance()

    IRREGULAR_PAST = {
        "be": "was",
        "is": "was",
        "are": "were",
        "bite": "bit",
        "run": "ran",
        "walk": "walked",
        "chase": "chased",
        "see": "saw",
        "hear": "heard",
        "think": "thought",
        "know": "knew",
        "say": "said",
        "tell": "told",
        "give": "gave",
        "take": "took",
        "go": "went",
        "come": "came",
        "eat": "ate",
        "drink": "drank",
        "have": "had",
        "has": "had",
        "make": "made",
        "build": "built",
        "find": "found",
        "leave": "left",
        "drop": "dropped",
        "pick": "picked",
        "move": "moved",
        "enter": "entered",
        "touch": "touched",
        "hit": "hit",
        "kill": "killed",
        "die": "died",
        "live": "lived",
        "want": "wanted",
        "feel": "felt",
        "grow": "grew",
        "happen": "happened",
    }

    IRREGULAR_PRES_3SG = {
        "be": "is",
        "have": "has",
        "do": "does",
        "go": "goes",
    }

    def realize_graph(self, graph: QuantaGraph) -> str:
        """Realizes an entire QuantaGraph ASG into an English sentence string without hardcoded templates."""
        root = graph.root
        if root is None:
            return ""

        # 1. Multi-sentence discourse / paragraph realization
        if root.anchor and root.anchor.startswith("discourse:"):
            token_nodes = [n.literal.strip() for n in graph._node_list if n.literal and isinstance(n.literal, str) and not (n.anchor and n.anchor.startswith("discourse:")) and len(n.literal.split()) <= 4]
            if len(token_nodes) >= 6:
                sentences = []
                curr_sent = []
                for tok in token_nodes:
                    curr_sent.append(tok)
                    if tok in (".", "!", "?"):
                        s_str = self._format_token_sequence(curr_sent)
                        if s_str:
                            sentences.append(s_str)
                        curr_sent = []
                if curr_sent:
                    s_str = self._format_token_sequence(curr_sent)
                    if s_str:
                        sentences.append(s_str)
                if sentences:
                    return " ".join(sentences)

            sub_clauses = []
            for child_cid in root.edges.get("GRAPH_IS_SUB_EXP", []):
                child_node = graph.get_node(child_cid)
                if child_node:
                    c_text = self._realize_clause_or_tokens(graph, child_node)
                    if c_text:
                        c_text = c_text.strip()
                        if not c_text.endswith((".", "?", "!")):
                            c_text += "."
                        sub_clauses.append(c_text[0].upper() + c_text[1:])
            if sub_clauses:
                return " ".join(sub_clauses)

        # 2. Check if word-level tokens exist in graph
        token_nodes = []
        for n in graph._node_list:
            if n.literal and isinstance(n.literal, str):
                lit = n.literal.strip()
                if not (n == root and len(lit.split()) > 4):
                    token_nodes.append(lit)

        if len(token_nodes) >= 3:
            return self._format_token_sequence(token_nodes)

        # 3. Check for conditional / implicational sentences
        if root.get_slot("LJB_GANAI_IF_THEN") == 1 or root.get_slot("GRAPH_BRANCH_COND") == 1:
            return self._realize_conditional(graph, root)

        # 4. Check if coordinating compound (and / or)
        if "GRAPH_IS_SUB_EXP" in root.edges and (root.get_slot("LJB_JE_AND") == 1 or root.get_slot("LJB_JA_OR") == 1):
            sub_clauses = []
            for child_cid in root.edges["GRAPH_IS_SUB_EXP"]:
                child_node = graph.get_node(child_cid)
                if child_node:
                    sub_clauses.append(self._realize_clause_or_tokens(graph, child_node))
            if len(sub_clauses) >= 2:
                c1 = sub_clauses[0].rstrip(".?!")
                c2 = sub_clauses[1].rstrip(".?!")
                conj = "and" if root.get_slot("LJB_JE_AND") == 1 else "or"
                return f"{c1[0].upper() + c1[1:]} {conj} {c2[0].lower() + c2[1:]}."

        # 5. Single sentence / clause realization
        clause_text = self._realize_clause(graph, root)
        if not clause_text:
            return ""

        # Capitalize and punctuate
        result = clause_text.strip()
        if not result.endswith((".", "?", "!")):
            if root.get_slot("GRAPH_QUERY_TARGET") == 3:
                result += "?"
            else:
                result += "."
        return result[0].upper() + result[1:]

    def _format_token_sequence(self, tokens: List[str]) -> str:
        """Formats a list of word/punct tokens into fluent English text with proper spacing and punctuation."""
        if not tokens:
            return ""
        text = ""
        prev_tok = ""
        for i, tok in enumerate(tokens):
            if not tok:
                continue
            # If literal was a composite like "did not see", avoid repeating "did not" if preceded by "did", "not"
            t_clean = tok
            if t_clean.startswith("did not ") and prev_tok == "not":
                t_clean = t_clean[8:]

            if i == 0 or not text:
                text = t_clean
            elif t_clean in (",", ".", ";", ":", "!", "?", "'s", "'ve", "'d", "'ll", "'re", "'m", "n't", "’s", "’ve", "’t", "n’t"):
                text += t_clean
            elif text.endswith("(") or t_clean in (")", "]", "}"):
                text += t_clean
            elif t_clean == "n't" or t_clean == "n’t":
                text += t_clean
            else:
                text += " " + t_clean
            prev_tok = t_clean

        if text:
            text = text[0].upper() + text[1:]
        return text

    def _collect_descendant_tokens(self, graph: QuantaGraph, root_node: QuantaNode) -> List[str]:
        """Collects and orders tokens belonging to a clause subgraph without traversing cross-sentence temporal edges."""
        visited_nodes = set()
        stack = [root_node]
        cross_sentence_rels = {"TEMP_ALLEN_MEETS", "TEMP_ALLEN_BEFORE", "TEMP_ALLEN_DURING", "TEMP_ALLEN_AFTER_INV", "GRAPH_ORDERED_SEQ"}
        while stack:
            curr = stack.pop()
            if curr in visited_nodes:
                continue
            visited_nodes.add(curr)
            for rel, targets in curr.edges.items():
                if rel in cross_sentence_rels:
                    continue
                for cid in targets:
                    child = graph.get_node(cid)
                    if child and child not in visited_nodes and not (child.anchor and child.anchor.startswith("discourse:")):
                        stack.append(child)

        # Filter and order tokens according to graph._node_list
        ordered_tokens = []
        for n in graph._node_list:
            if n in visited_nodes and n.literal and isinstance(n.literal, str):
                lit = n.literal.strip()
                if len(lit.split()) <= 4:
                    ordered_tokens.append(lit)

        return ordered_tokens

    def _realize_clause_or_tokens(self, graph: QuantaGraph, node: QuantaNode) -> str:
        """Realizes a sentence/clause using word-level tokens when present, falling back to compositional SVO unrolling."""
        # For multi-sentence discourse graphs, collect tokens for the specific clause node
        if graph.root and graph.root.anchor and graph.root.anchor.startswith("discourse:"):
            ordered_tokens = self._collect_descendant_tokens(graph, node)
            if len(ordered_tokens) >= 3:
                return self._format_token_sequence(ordered_tokens)
            return self._realize_clause(graph, node)

        # For single sentence graphs, use the sequence of word tokens in the graph
        token_nodes = []
        for n in graph._node_list:
            if n.literal and isinstance(n.literal, str):
                lit = n.literal.strip()
                if not (n == node and len(lit.split()) > 4):
                    token_nodes.append(lit)

        if len(token_nodes) >= 3:
            return self._format_token_sequence(token_nodes)

        return self._realize_clause(graph, node)

    def _realize_conditional(self, graph: QuantaGraph, root: QuantaNode) -> str:
        """Realizes conditional/implication structures (If A, then B)."""
        cond_node = None
        then_node = None

        if "GRAPH_BRANCH_COND" in root.edges and root.edges["GRAPH_BRANCH_COND"]:
            cond_node = graph.get_node(root.edges["GRAPH_BRANCH_COND"][0])
        if "GRAPH_BRANCH_THEN" in root.edges and root.edges["GRAPH_BRANCH_THEN"]:
            then_node = graph.get_node(root.edges["GRAPH_BRANCH_THEN"][0])

        if not cond_node and "GRAPH_IS_SUB_EXP" in root.edges:
            children = [graph.get_node(cid) for cid in root.edges["GRAPH_IS_SUB_EXP"]]
            if len(children) >= 2:
                cond_node, then_node = children[0], children[1]

        if cond_node and then_node:
            cond_str = self._realize_clause(graph, cond_node)
            then_str = self._realize_clause(graph, then_node)
            return f"If {cond_str}, then {then_str}."

        # Fallback if unlinked condition
        return self._realize_clause(graph, root) + "."

    def _realize_clause(self, graph: QuantaGraph, predicate_node: QuantaNode) -> str:
        """Realizes a single predicate-argument clause (SVO + Modifiers)."""
        # 1. Resolve Verb Base & Inflection
        verb_base = self._extract_verb_base(predicate_node)

        is_past = predicate_node.get_slot("LJB_PU_PAST_TENSE") == 1 or \
                  predicate_node.get_slot("LJB_ZI_SHORT_PAST") == 1 or \
                  predicate_node.get_slot("LJB_ZA_MEDIUM_PAST") == 1 or \
                  predicate_node.get_slot("LJB_ZU_LONG_PAST") == 1
        is_future = predicate_node.get_slot("LJB_BA_FUTURE_TENSE") == 1
        is_negated = predicate_node.get_slot("LJB_NA_NEGATION") == 2

        # Modals
        is_obligation = predicate_node.get_slot("EPIST_DEONTIC_OBLIGATION") == 1
        is_prohibition = predicate_node.get_slot("EPIST_DEONTIC_PROHIBITION") == 1
        is_permission = predicate_node.get_slot("EPIST_DEONTIC_PERMISSION") == 1
        is_possibility = predicate_node.get_slot("NSM_MAYBE") == 3 or predicate_node.get_slot("MODALITY_HYPOTHETICAL") == 3
        is_probable = predicate_node.get_slot("EPIST_PROB_HIGH") == 1

        # 2. Resolve Subject / Agent (VAL_X1_AGENT)
        agent_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X1_AGENT", default_role="someone")

        # 3. Form Verb Phrase
        verb_phrase = self._form_verb_phrase(
            verb_base=verb_base,
            is_past=is_past,
            is_future=is_future,
            is_negated=is_negated,
            is_obligation=is_obligation,
            is_prohibition=is_prohibition,
            is_permission=is_permission,
            is_possibility=is_possibility,
            is_probable=is_probable,
            subject=agent_str,
        )

        # 4. Resolve Patient / Object (VAL_X2_PATIENT) or Experiencer
        patient_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X2_PATIENT")
        experiencer_str = ""
        if "VAL_EXPERIENCER" in predicate_node.edges:
            if patient_str or verb_base in ("run", "walk", "go", "come", "move"):
                experiencer_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_EXPERIENCER", prep="to")
            else:
                patient_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_EXPERIENCER")

        # 5. Resolve Prepositional Arguments
        dest_cids = predicate_node.edges.get("VAL_X3_DESTINATION", [])
        dest_node = graph.get_node(dest_cids[0]) if dest_cids else None
        dest_prep = "into" if dest_node and dest_node.get_slot("NSM_INSIDE") == 1 else "to"
        dest_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X3_DESTINATION", prep=dest_prep)
        source_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X4_SOURCE", prep="from")
        inst_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X5_INSTRUMENT", prep="with")
        loc_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_LOCATION_SLOT", prep="in")
        purpose_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_PURPOSE_SLOT", prep="for")

        # 6. Resolve Temporal, Manner, & Predicate Adjective Phrases
        time_str = self._resolve_temporal_phrase(predicate_node)
        manner_str = self._resolve_manner_phrase(predicate_node)

        pred_adj = ""
        is_copula = verb_base in ("be", "is", "are", "was", "were")
        if is_copula:
            if predicate_node.get_slot("NSM_FEEL") == 1 and predicate_node.get_slot("NSM_GOOD") == 1:
                pred_adj = "happy"
            elif predicate_node.get_slot("NSM_BIG") == 1:
                pred_adj = "big"
            elif predicate_node.get_slot("NSM_SMALL") == 1:
                pred_adj = "small"
            elif predicate_node.get_slot("NSM_GOOD") == 1:
                pred_adj = "good"
            elif predicate_node.get_slot("NSM_BAD") == 1:
                pred_adj = "bad"
            elif predicate_node.get_slot("NSM_FEEL") == 1 and predicate_node.get_slot("TYPE_ATTRIBUTE_PROPERTY") == 1:
                pred_adj = "cold"
            elif predicate_node.get_slot("NSM_TOUCH") == 1 and predicate_node.get_slot("TYPE_ATTRIBUTE_PROPERTY") == 1:
                pred_adj = "hard"

        # 7. Check if Interrogative / Question
        is_query = predicate_node.get_slot("GRAPH_QUERY_TARGET") == 3
        if is_query:
            is_copula = verb_base in ("be", "is", "are")
            aux = ""
            modal_adv = ""
            main_verb = verb_base

            if is_possibility:
                modal_adv = "perhaps"

            if is_obligation:
                aux = "must"
            elif is_permission:
                aux = "can"
            elif is_future:
                aux = "will"
            elif is_past:
                if is_copula:
                    aux = "was"
                    main_verb = ""
                else:
                    aux = "did"
            else:  # present
                if is_copula:
                    aux = "is"
                    main_verb = ""
                else:
                    aux = "does"

            if is_negated:
                aux = f"{aux} not" if aux else "not"

            tokens = [aux, agent_str]
            if modal_adv:
                tokens.append(modal_adv)
            if main_verb:
                tokens.append(main_verb)
            if pred_adj:
                tokens.append(pred_adj)
            if manner_str:
                tokens.append(manner_str)
            if patient_str:
                tokens.append(patient_str)
            if experiencer_str:
                tokens.append(experiencer_str)
            if dest_str:
                tokens.append(dest_str)
            if source_str:
                tokens.append(source_str)
            if inst_str:
                tokens.append(inst_str)
            if loc_str:
                tokens.append(loc_str)
            if purpose_str:
                tokens.append(purpose_str)
            if time_str:
                tokens.append(time_str)

            return " ".join(t for t in tokens if t).strip()

        # Assemble clause tokens in canonical SVO order
        tokens = [agent_str] if agent_str else []
        if verb_phrase:
            tokens.append(verb_phrase)
        if pred_adj:
            tokens.append(pred_adj)
        if manner_str:
            tokens.append(manner_str)
        if patient_str:
            tokens.append(patient_str)
        if experiencer_str:
            tokens.append(experiencer_str)
        if dest_str:
            tokens.append(dest_str)
        if source_str:
            tokens.append(source_str)
        if inst_str:
            tokens.append(inst_str)
        if loc_str:
            tokens.append(loc_str)
        if purpose_str:
            tokens.append(purpose_str)
        if time_str:
            tokens.append(time_str)

        return " ".join(t for t in tokens if t).strip()

    def _extract_verb_base(self, node: QuantaNode) -> str:
        """Extracts the base verb lemma from node anchor, literal, or active NSM prime."""
        if node.anchor:
            if node.anchor.startswith("cn:"):
                anc = node.anchor[3:]
                if ":" in anc:
                    anc = anc.split(":", 1)[1]
                if " (" in anc:
                    anc = anc.split(" (")[0]
                return anc.replace("_", " ").strip()
            elif node.anchor.startswith("wn:"):
                # e.g., 'wn:bite.v.01' -> 'bite'
                parts = node.anchor[3:].split(".")
                if len(parts) > 0 and parts[0]:
                    return parts[0].replace("_", " ")

        if node.literal and isinstance(node.literal, str):
            lit = node.literal.strip()
            # If literal is a simple verb or short text
            if len(lit.split()) == 1 and lit.isalpha():
                return lit.lower()

        # Vector decoding fallback if Band 3/4 non-zero
        if hasattr(node, "vector") and node.vector is not None:
            vec_data = node.vector._data if hasattr(node.vector, "_data") else None
            if vec_data is not None and np.any(vec_data[384:640]):
                lemma, dist = self.vector_decoder.decode_vector(vec_data[384:640])
                if lemma and dist <= 0.15:
                    return lemma

        # Fallback: Infer from Band 0 NSM primes
        if node.get_slot("NSM_MOVE") != 0:
            return "move"
        if node.get_slot("NSM_SAY") != 0:
            return "say"
        if node.get_slot("NSM_THINK") != 0:
            return "think"
        if node.get_slot("NSM_KNOW") != 0:
            return "know"
        if node.get_slot("NSM_WANT") != 0:
            return "want"
        if node.get_slot("NSM_FEEL") != 0:
            return "feel"
        if node.get_slot("NSM_SEE") != 0:
            return "see"
        if node.get_slot("NSM_HEAR") != 0:
            return "hear"
        if node.get_slot("NSM_TOUCH") != 0:
            return "touch"
        if node.get_slot("NSM_DO") != 0:
            return "do"
        if node.get_slot("NSM_LIVE") != 0:
            return "live"
        if node.get_slot("NSM_DIE") != 0:
            return "die"
        if node.get_slot("NSM_GROW") != 0:
            return "grow"
        if node.get_slot("NSM_HAPPEN") != 0:
            return "happen"
        if node.get_slot("NSM_BE_SOMEWHERE") != 0 or node.get_slot("LJB_DU_IDENTITY") == 1:
            return "be"

        return "is"

    def _form_verb_phrase(
        self,
        verb_base: str,
        is_past: bool,
        is_future: bool,
        is_negated: bool,
        is_obligation: bool,
        is_permission: bool,
        is_possibility: bool,
        is_probable: bool,
        subject: str = "",
        is_prohibition: bool = False,
    ) -> str:
        """Constructs an inflected verb phrase with tense, modals, and negation."""
        is_copula = verb_base in ("be", "is", "are")
        
        # Modal auxiliary construction
        if is_obligation or is_prohibition:
            modal = "must not" if (is_negated or is_prohibition) else "must"
            return f"{modal} {verb_base if not is_copula else 'be'}"
        if is_permission:
            modal = "cannot" if is_negated else "can"
            return f"{modal} {verb_base if not is_copula else 'be'}"
        if is_possibility:
            modal = "might not" if is_negated else "might"
            return f"{modal} {verb_base if not is_copula else 'be'}"
        if is_future:
            modal = "will not" if is_negated else "will"
            return f"{modal} {verb_base if not is_copula else 'be'}"

        prob_prefix = "probably " if is_probable else ""

        # Past tense
        if is_past:
            if is_negated:
                if is_copula:
                    return f"{prob_prefix}was not"
                return f"{prob_prefix}did not {verb_base}"
            past_form = self.IRREGULAR_PAST.get(verb_base)
            if not past_form:
                if verb_base.endswith("e"):
                    past_form = verb_base + "d"
                elif verb_base.endswith("y") and len(verb_base) > 1 and verb_base[-2] not in "aeiou":
                    past_form = verb_base[:-1] + "ied"
                else:
                    past_form = verb_base + "ed"
            return f"{prob_prefix}{past_form}"

        # Present tense
        if is_negated:
            if is_copula:
                return f"{prob_prefix}is not"
            return f"{prob_prefix}does not {verb_base}"

        if is_copula:
            return f"{prob_prefix}is"

        pres_form = self.IRREGULAR_PRES_3SG.get(verb_base)
        if not pres_form:
            if verb_base.endswith(("s", "sh", "ch", "x", "z", "o")):
                pres_form = verb_base + "es"
            elif verb_base.endswith("y") and len(verb_base) > 1 and verb_base[-2] not in "aeiou":
                pres_form = verb_base[:-1] + "ies"
            else:
                pres_form = verb_base + "s"
        return f"{prob_prefix}{pres_form}"

    def _resolve_entity_by_edge(
        self,
        graph: QuantaGraph,
        node: QuantaNode,
        relation: str,
        prep: Optional[str] = None,
        default_role: Optional[str] = None,
    ) -> str:
        """Resolves target node of relation edge into an entity noun phrase."""
        if relation not in node.edges or not node.edges[relation]:
            return default_role or ""

        target_cid = node.edges[relation][0]
        target_node = graph.get_node(target_cid)
        if target_node is None:
            return default_role or ""

        np = self._realize_noun_phrase(target_node)
        if prep and np:
            return f"{prep} {np}"
        return np

    def _realize_noun_phrase(self, node: QuantaNode) -> str:
        """Realizes an entity node with determiners, quantifiers, descriptors, and head noun."""
        head_noun = ""

        if node.anchor:
            if node.anchor.startswith("cn:"):
                anc = node.anchor[3:]
                if ":" in anc:
                    anc = anc.split(":", 1)[1]
                if " (" in anc:
                    anc = anc.split(" (")[0]
                head_noun = anc.replace("_", " ").strip()
            elif node.anchor.startswith("wn:"):
                # e.g., 'wn:golden_retriever.n.01' -> 'golden retriever'
                parts = node.anchor[3:].split(".")
                if len(parts) > 0:
                    head_noun = parts[0].replace("_", " ")

        if not head_noun and node.literal:
            lit = str(node.literal).strip()
            # If literal is a clean entity name
            head_noun = lit

        # Vector decoding fallback for anchor-free nodes
        if not head_noun and hasattr(node, "vector") and node.vector is not None:
            vec_data = node.vector._data if hasattr(node.vector, "_data") else None
            if vec_data is not None and np.any(vec_data[384:640]):
                lemma, dist = self.vector_decoder.decode_vector(vec_data[384:640])
                if lemma and dist <= 0.15:
                    head_noun = lemma

        if head_noun in ("homo", "homo sapiens", "human being"):
            if node.literal and str(node.literal).lower() in ("human", "person", "man", "woman"):
                head_noun = str(node.literal).lower()
            else:
                head_noun = "human"
        elif node.literal and isinstance(node.literal, str):
            lit = str(node.literal).strip().lower()
            if lit in ("human", "person", "dog", "cat", "mailman", "rock", "garden", "house", "car", "tree", "water", "book", "city", "stick", "ball"):
                head_noun = lit

        if not head_noun:
            if node.get_slot("TYPE_HUMAN") == 1 or node.get_slot("CN_Q015_PERSON") == 1:
                head_noun = "person"
            elif node.get_slot("TYPE_ANIMATE") == 1 or node.get_slot("CN_Q011_ANIMAL") == 1:
                head_noun = "animal"
            elif node.get_slot("TYPE_ARTIFACT") == 1 or node.get_slot("CN_Q042_DEVICE") == 1:
                head_noun = "object"
            elif node.get_slot("TYPE_SPATIAL_REGION") == 1 or node.get_slot("CN_Q073_PLACE") == 1:
                head_noun = "place"
            elif node.get_slot("TYPE_ABSTRACT_CONCEPT") == 1 or node.get_slot("CN_Q113_LOGIC") == 1:
                head_noun = "concept"
            else:
                head_noun = "entity"

        # Check if head noun is a proper noun or pronoun
        if head_noun.istitle() and len(head_noun.split()) == 1 and head_noun.lower() not in ("person", "human", "animal", "dog", "cat", "mailman"):
            return head_noun
        if head_noun.lower() in ("i", "you", "he", "she", "it", "we", "they", "someone", "something", "everyone", "nothing"):
            return head_noun.lower()

        # Descriptors / Adjectives
        adjectives = []
        if node.get_slot("NSM_BIG") == 1:
            adjectives.append("big")
        if node.get_slot("NSM_SMALL") == 1:
            adjectives.append("small")
        if node.get_slot("NSM_GOOD") == 1:
            adjectives.append("good")
        if node.get_slot("NSM_BAD") == 1:
            adjectives.append("bad")
        if node.get_slot("NSM_FEEL") == 1:
            adjectives.append("cold")
        if node.get_slot("NSM_TOUCH") == 1:
            adjectives.append("hard")

        # Quantifiers & Determiners
        determiner = ""
        if node.get_slot("LJB_RO_ALL_QUANT") == 1 or node.get_slot("NSM_ALL") == 1:
            determiner = "every"
        elif node.get_slot("LJB_NO_NONE_QUANT") == 1:
            determiner = "no"
        elif node.get_slot("NSM_TWO") == 1 or node.get_slot("LJB_MEI_CARDINAL") == 1:
            determiner = "two"
        elif node.get_slot("NSM_SOME") == 1 or node.get_slot("LJB_SUO_AT_LEAST_ONE") == 1:
            determiner = "a"
        elif node.get_slot("NSM_THIS") == 1:
            determiner = "the"
        elif node.get_slot("NSM_OTHER") == 1:
            determiner = "another"
        else:
            determiner = "a"

        # Correct 'a' vs 'an'
        desc_str = " ".join(adjectives)
        first_word = adjectives[0] if adjectives else head_noun
        if determiner == "a" and first_word and first_word[0].lower() in "aeiou":
            determiner = "an"

        tokens = [determiner] if determiner else []
        if adjectives:
            tokens.extend(adjectives)
        tokens.append(head_noun)

        return " ".join(tokens)

    def _resolve_temporal_phrase(self, node: QuantaNode) -> str:
        """Extracts temporal adverbials from Band 0 & Band 1 slots."""
        if node.get_slot("NSM_NOW") == 1:
            return "now"
        if node.get_slot("NSM_BEFORE") == 1:
            if node.get_slot("LJB_ZI_SHORT_PAST") == 1:
                return "recently"
            if node.get_slot("LJB_ZU_LONG_PAST") == 1:
                return "long ago"
            return "earlier"
        if node.get_slot("NSM_AFTER") == 1:
            return "later"
        if node.get_slot("NSM_A_SHORT_TIME") == 1:
            return "soon"
        if node.get_slot("NSM_A_LONG_TIME") == 1:
            return "always"
        return ""

    def _resolve_manner_phrase(self, node: QuantaNode) -> str:
        """Extracts manner adverbials."""
        if node.get_slot("NSM_ACCELERATING_RATE") == 1:
            return "rapidly"
        if node.get_slot("NSM_CONTINUOUS_RATE") == 1:
            return "continuously"
        return ""
