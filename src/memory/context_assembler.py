"""Bipartite Passage Projection & Realization Bypass (Context Assembly) for QUANTA SVM.

Implements Section 7 of QUANTA SVM Refactor Master Plan:
1. BipartiteProjector: Projects semantic activation scores p(v_s) from ASG nodes (V_semantic)
   onto raw passage nodes (V_passage) in PassageStore:
       Score(v_p) = sum_{v_s in N(v_p)} p(v_s) * conf(v_s)
   Ranks passages by accumulated score and enforces token budgets.
2. DualStreamContextAssembler: Bypasses lossy neural NLG realization to construct a dual-stream
   context for downstream reader LLMs:
   - Stream 1: Logical Briefing Block (verified causal paths, Allen intervals, thematic valency frames)
   - Stream 2: Top-K Raw Source Passages (verbatim text spans from PassageStore with 100% lexical fidelity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from memory.passage_store import PassageRecord, PassageStore

logger = logging.getLogger("quanta.memory.context_assembler")


@dataclass(slots=True)
class ProjectedPassage:
    """Projected passage node (V_passage) with accumulated semantic activation score.

    Attributes:
        passage_id: Unique passage identifier in PassageStore (e.g. 'P104').
        doc_id: Source document or chapter identifier.
        text: Pristine verbatim text content of the passage.
        score: Accumulated bipartite projected activation score.
        char_span: Absolute character range (start, end) within source document.
        contributing_node_cids: CIDs of semantic nodes that projected score onto this passage.
        activated_spans: Character spans within passage or document corresponding to active nodes.
        metadata: Additional provenance and scoring metadata.
    """

    passage_id: str
    doc_id: str
    text: str
    score: float
    char_span: Tuple[int, int] = (0, 0)
    contributing_node_cids: List[str] = field(default_factory=list)
    activated_spans: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.char_span, list):
            object.__setattr__(self, "char_span", tuple(self.char_span))


@dataclass
class DualStreamContext:
    """Structured container holding dual-stream context representation.

    Attributes:
        logical_briefing: Stream 1 (Logical Briefing Block).
        passage_stream: Stream 2 (Retrieved Top-K Raw Source Passages).
        full_context: Complete joined context string ready for LLM prompt injection.
        passages: List of ProjectedPassage objects included in the passage stream.
        token_count_estimate: Estimated token count of the assembled context.
    """

    logical_briefing: str
    passage_stream: str
    full_context: str
    passages: List[ProjectedPassage] = field(default_factory=list)
    token_count_estimate: int = 0


# =============================================================================
# Helper Utilities
# =============================================================================

def _clean_node_label(node: QuantaNode) -> str:
    """Extracts a clean, human-readable surface label or predicate for a QuantaNode."""
    if node.literal is not None:
        if isinstance(node.literal, dict):
            lbl = node.literal.get("label") or node.literal.get("name") or node.literal.get("predicate")
            if lbl:
                return str(lbl).strip()
        elif isinstance(node.literal, str):
            lit = node.literal.strip()
            # If literal contains a full passage or long paragraph, return anchor or first words
            if len(lit.split()) <= 6 and not lit.startswith("{"):
                return lit
            if node.anchor:
                anchor_clean = node.anchor
                if anchor_clean.startswith("cn:en:"):
                    anchor_clean = anchor_clean[6:]
                anchor_clean = re.sub(r"\s*\([nav]\)$", "", anchor_clean)
                return anchor_clean.replace("_", " ").strip()
            return lit.split()[:4] and " ".join(lit.split()[:4]) + "..." or lit
        else:
            return str(node.literal)

    if node.anchor:
        clean = node.anchor
        if clean.startswith("cn:en:"):
            clean = clean[6:]
        clean = re.sub(r"\s*\([nav]\)$", "", clean)
        clean = clean.replace("_", " ")
        return clean.strip()

    return node.cid[:8]


def _format_time_interval(node: QuantaNode) -> str:
    """Formats temporal interval bounds for an event node."""
    t_start = getattr(node, "time_start", None)
    t_end = getattr(node, "time_end", None)
    if t_start is not None and t_end is not None:
        return f"[{t_start}..{t_end}]"
    elif t_start is not None:
        return f"[{t_start}..]"
    elif t_end is not None:
        return f"[..{t_end}]"
    return ""


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Cleanly truncates text to fit within token budget at sentence or paragraph boundaries."""
    if not text:
        return ""
    max_words = max(1, int(max_tokens / 1.33))
    words = text.split()
    if len(words) <= max_words:
        return text

    # Find character cutoff corresponding to max_words
    word_count = 0
    char_cutoff = len(text)
    for m in re.finditer(r"\S+", text):
        word_count += 1
        if word_count >= max_words:
            char_cutoff = m.end()
            break

    candidate = text[:char_cutoff]

    # 1. Prefer truncating at double newline (paragraph boundary)
    last_para = candidate.rfind("\n\n")
    if last_para > len(candidate) * 0.5:
        return candidate[:last_para].strip()

    # 2. Prefer truncating at sentence boundary
    last_punct = max(
        candidate.rfind(".\n"), candidate.rfind(". "),
        candidate.rfind("!\n"), candidate.rfind("? ")
    )
    if last_punct > len(candidate) * 0.5:
        return candidate[:last_punct + 1].strip()

    last_punct_any = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
    if last_punct_any > len(candidate) * 0.5:
        return candidate[:last_punct_any + 1].strip()

    # 3. Fallback to newline or word boundary
    last_newline = candidate.rfind("\n")
    if last_newline > len(candidate) * 0.5:
        return candidate[:last_newline].strip()

    res = candidate.strip()
    if not res.endswith((".", "!", "?")):
        res += "."
    return res


# =============================================================================
# BipartiteProjector
# =============================================================================

class BipartiteProjector:
    """Projects semantic activation scores from ASG nodes (V_semantic) onto passage nodes (V_passage).

    Mathematical Formulation:
        Score(v_p) = sum_{v_s in N(v_p)} p(v_s) * conf(v_s)
    where:
        - v_p is a raw source passage in PassageStore
        - N(v_p) is the set of semantic concept/event nodes grounded in v_p
        - p(v_s) is the spreading activation / PPR score of semantic node v_s
        - conf(v_s) is the provenance confidence of v_s
    """

    def __init__(
        self,
        passage_store: Optional[PassageStore] = None,
        min_score_threshold: float = 0.0,
        filter_contradictions: bool = True,
    ):
        """Initializes BipartiteProjector.

        Args:
            passage_store: Optional default PassageStore containing raw passage records.
            min_score_threshold: Minimum projected score required to retain a passage.
            filter_contradictions: If True, prunes nodes marked with Belnap CONTRADICTION (00_2).
        """
        self.passage_store = passage_store
        self.min_score_threshold = float(min_score_threshold)
        self.filter_contradictions = bool(filter_contradictions)

    def project(
        self,
        subgraph: QuantaGraph,
        passage_store: Optional[PassageStore] = None,
        activations: Optional[Dict[str, float]] = None,
    ) -> List[ProjectedPassage]:
        """Projects semantic node activations onto passage nodes and ranks passages by score.

        Args:
            subgraph: Retrieved QuantaGraph ASG sub-graph.
            passage_store: Optional PassageStore (overrides default).
            activations: Optional dict of node CID -> activation score (defaults to graph activations or node salience).

        Returns:
            List of ProjectedPassage instances sorted by descending projected score.
        """
        store = passage_store or self.passage_store
        act_map: Dict[str, float] = {}
        if activations is not None:
            act_map = activations
        elif hasattr(subgraph, "activations") and isinstance(getattr(subgraph, "activations"), dict):
            act_map = getattr(subgraph, "activations")

        passage_scores: Dict[str, float] = {}
        passage_nodes: Dict[str, List[QuantaNode]] = {}
        passage_spans: Dict[str, List[Tuple[int, int]]] = {}

        # 1. Iterate over all semantic nodes in subgraph
        for cid, node in subgraph.nodes.items():
            # Check Belnap contradiction filtering under PoP-RAG
            truth = getattr(node, "truth_status", "TRUE")
            if self.filter_contradictions and truth in ("CONTRADICTION", "00", "00_2"):
                continue

            # Determine activation p(v_s) and confidence conf(v_s)
            p_val = act_map.get(
                cid,
                act_map.get(getattr(node, "canonical_cid", ""), getattr(node, "salience", 1.0))
            )
            conf = getattr(node, "confidence", 1.0)
            weight = float(p_val * conf)

            # Discover all passage IDs grounded to this semantic node
            grounded_pids: Set[str] = set()

            # Direct node attribute
            if getattr(node, "passage_id", None):
                grounded_pids.add(str(node.passage_id))

            # Graph node_to_passage mapping
            if hasattr(subgraph, "node_to_passage") and cid in subgraph.node_to_passage:
                grounded_pids.add(str(subgraph.node_to_passage[cid][0]))

            # Graph passage_to_nodes mapping
            if hasattr(subgraph, "passage_to_nodes"):
                for p_id, cids in subgraph.passage_to_nodes.items():
                    if cid in cids:
                        grounded_pids.add(str(p_id))

            # Bipartite relation edges (E_ground)
            if hasattr(node, "edges") and isinstance(node.edges, dict):
                for rel in ("E_GROUND", "GROUNDED_IN", "PASSAGE"):
                    if rel in node.edges:
                        for tgt in node.edges[rel]:
                            grounded_pids.add(str(tgt))

            # Legacy fallback: fold nodes or document headings in parent_cid
            if not grounded_pids:
                p_cid = getattr(node, "parent_cid", None)
                if p_cid and re.match(r"^(?:Document|Passage|chapter|doc_)\w*", str(p_cid), re.IGNORECASE):
                    grounded_pids.add(str(p_cid).strip())

            # Accumulate bipartite scores
            for pid in grounded_pids:
                passage_scores[pid] = passage_scores.get(pid, 0.0) + weight
                passage_nodes.setdefault(pid, []).append(node)
                s_start = getattr(node, "span_start", None)
                s_end = getattr(node, "span_end", None)
                if s_start is not None and s_end is not None:
                    passage_spans.setdefault(pid, []).append((int(s_start), int(s_end)))

        # Fallback for episodic fold nodes when PassageStore is empty or unpopulated
        fold_nodes = [
            n for n in subgraph.nodes.values()
            if (n.anchor and (n.anchor.startswith("merkle:") or n.anchor.startswith("fold:")))
            or (isinstance(n.literal, dict) and n.literal.get("type") == "discourse_episode_fold")
        ]
        fallback_texts: Dict[str, str] = {}
        for fn in fold_nodes:
            if isinstance(fn.literal, dict):
                p_text = fn.literal.get("text")
                d_tag = fn.parent_cid or fn.literal.get("chapter_title") or fn.cid[:8]
                if p_text:
                    fallback_texts[str(d_tag)] = str(p_text)
                    if d_tag not in passage_scores:
                        fn_act = act_map.get(fn.cid, getattr(fn, "salience", 1.0))
                        passage_scores[str(d_tag)] = float(fn_act)
                        passage_nodes.setdefault(str(d_tag), []).append(fn)

        # 2. Materialize ProjectedPassage records
        projected_passages: List[ProjectedPassage] = []

        for pid, total_score in passage_scores.items():
            if total_score < self.min_score_threshold:
                continue

            # Retrieve pristine raw text from PassageStore
            text = ""
            doc_id = "default_doc"
            char_span = (0, 0)

            rec: Optional[PassageRecord] = None
            if store is not None:
                rec = store.get_passage(pid)

            if rec is not None:
                text = rec.text
                doc_id = rec.doc_id
                char_span = rec.char_span
            elif pid in fallback_texts:
                text = fallback_texts[pid]
                doc_id = pid
                char_span = (0, len(text))
            elif store is not None and hasattr(store, "get_passages_for_doc"):
                doc_passages = store.get_passages_for_doc(pid)
                if doc_passages:
                    combined = " ".join(p.text for p in doc_passages)
                    text = combined
                    doc_id = pid
                    char_span = (0, len(combined))

            contributing = [n.cid for n in passage_nodes.get(pid, [])]
            spans = passage_spans.get(pid, [])

            projected_passages.append(
                ProjectedPassage(
                    passage_id=pid,
                    doc_id=doc_id,
                    text=text,
                    score=total_score,
                    char_span=char_span,
                    contributing_node_cids=contributing,
                    activated_spans=spans,
                )
            )

        # 3. Rank passages by descending projected score (ties broken by passage length)
        projected_passages.sort(key=lambda p: (p.score, len(p.text)), reverse=True)
        return projected_passages

    def select_top_passages(
        self,
        ranked_passages: List[ProjectedPassage],
        max_tokens: int,
    ) -> List[ProjectedPassage]:
        """Greedily selects top-ranked passages within the specified token budget.

        Preserves 100% lexical fidelity of included passage spans.

        Args:
            ranked_passages: List of ProjectedPassage sorted by score descending.
            max_tokens: Maximum token budget for passage stream.

        Returns:
            List of selected ProjectedPassage instances.
        """
        if not ranked_passages or max_tokens <= 0:
            return []

        selected: List[ProjectedPassage] = []
        accumulated_tokens = 0

        for p in ranked_passages:
            # Estimate passage tokens (~1.33 tokens per word + 10 tokens header overhead)
            p_words = len(p.text.split())
            p_tokens = max(1, int(p_words * 1.33)) + 10

            if accumulated_tokens + p_tokens <= max_tokens:
                selected.append(p)
                accumulated_tokens += p_tokens
            elif not selected:
                # If first passage alone exceeds budget, include it (caller can truncate text cleanly)
                selected.append(p)
                break
            else:
                # Token budget reached
                break

        return selected


# =============================================================================
# DualStreamContextAssembler
# =============================================================================

class DualStreamContextAssembler:
    """Assembles dual-stream retrieval context for downstream reader LLMs.

    Stream 1: Logical Briefing Block
        - Verified active causal paths and temporal intervals
        - Thematic valency frames (S-V-O roles, instruments, locations)
        - Epistemic evidence source and speech-act intent
        - Key ontological relations

    Stream 2: Top-K Raw Source Passages
        - Pristine verbatim lexical text spans extracted directly from PassageStore
    """

    def __init__(
        self,
        projector: Optional[BipartiteProjector] = None,
        passage_store: Optional[PassageStore] = None,
        logical_ratio: float = 0.25,
        passage_ratio: float = 0.75,
    ):
        """Initializes DualStreamContextAssembler.

        Args:
            projector: Optional BipartiteProjector instance.
            passage_store: Optional PassageStore for raw text lookups.
            logical_ratio: Fraction of token budget reserved for Stream 1 (default 0.25).
            passage_ratio: Fraction of token budget reserved for Stream 2 (default 0.75).
        """
        self.passage_store = passage_store
        self.projector = projector or BipartiteProjector(passage_store=self.passage_store)
        self.logical_ratio = logical_ratio
        self.passage_ratio = passage_ratio

    def assemble_logical_briefing(
        self,
        subgraph: QuantaGraph,
        max_tokens: int = 300,
    ) -> str:
        """Constructs Stream 1: Logical Briefing Block from active ASG sub-graph.

        Summarizes:
        - Verified causal paths (CAUSAL_MECHANISM_LINK, CAUSAL_LEADS_TO)
        - Temporal intervals (TEMP_ALLEN_MEETS, TEMP_ALLEN_BEFORE, etc.)
        - Thematic valency frames (Agent, Patient, Location, Instrument)
        - Epistemic evidence sources and speech-act intent

        Args:
            subgraph: Active QuantaGraph sub-graph.
            max_tokens: Token budget for the logical briefing.

        Returns:
            Formatted Logical Briefing Block string.
        """
        if not subgraph.nodes:
            return ""

        causal_temporal_lines: List[str] = []
        valency_lines: List[str] = []
        epistemic_lines: List[str] = []
        relational_lines: List[str] = []
        entity_names: List[str] = []

        seen_causal: Set[Tuple[str, str, str]] = set()
        seen_valencies: Set[str] = set()

        # Identify event nodes vs entity nodes
        for cid, node in subgraph.nodes.items():
            if node.anchor and (node.anchor.startswith("merkle:") or node.anchor.startswith("fold:")):
                continue

            is_event = (
                node.get_slot("TYPE_EVENT") == 1
                or node.get_slot("WN_ACT_ACTION") == 1
                or getattr(node, "node_type", None) == "EVENT"
                or (node.anchor and "(v)" in node.anchor)
            )

            lbl = _clean_node_label(node)

            if is_event:
                ev_id = f"EV_{cid[:6]}"
                t_str = _format_time_interval(node)
                ev_disp = f"{ev_id} ({lbl}) {t_str}".strip() if t_str else f"{ev_id} ({lbl})"

                # 1. Causal & Temporal Paths
                if hasattr(node, "edges") and isinstance(node.edges, dict):
                    for rel, targets in node.edges.items():
                        if rel in (
                            "CAUSAL_MECHANISM_LINK", "CAUSAL_LEADS_TO",
                            "TEMP_ALLEN_MEETS", "TEMP_ALLEN_BEFORE",
                            "TEMP_ALLEN_OVERLAPS", "TEMP_ALLEN_DURING",
                        ):
                            for t_cid in targets:
                                t_node = subgraph.get_node(t_cid)
                                t_lbl = _clean_node_label(t_node) if t_node else t_cid[:6]
                                t_time = _format_time_interval(t_node) if t_node else ""
                                t_disp = f"EV_{t_cid[:6]} ({t_lbl}) {t_time}".strip() if t_time else f"EV_{t_cid[:6]} ({t_lbl})"

                                key = (cid, rel, t_cid)
                                if key not in seen_causal:
                                    seen_causal.add(key)
                                    status = getattr(node, "truth_status", "Verified")
                                    conf = getattr(node, "confidence", 1.0)
                                    line = f"- Path: {ev_disp} -> {rel} -> {t_disp} | Status: {status} (conf={conf:.2f})."
                                    causal_temporal_lines.append(line)

                # 2. Thematic Valency Frames
                frame_roles: List[str] = []
                role_mappings = [
                    ("VAL_X1_AGENT", "Agent"),
                    ("VAL_X2_PATIENT", "Patient"),
                    ("VAL_LOCATION_SLOT", "Location"),
                    ("VAL_X5_INSTRUMENT", "Instrument"),
                ]
                for rel_key, role_name in role_mappings:
                    if rel_key in node.edges and node.edges[rel_key]:
                        role_ents = []
                        for ent_cid in node.edges[rel_key]:
                            ent_node = subgraph.get_node(ent_cid)
                            role_ents.append(_clean_node_label(ent_node) if ent_node else ent_cid[:6])
                        if role_ents:
                            frame_roles.append(f"{role_name}: {', '.join(role_ents)}")

                if frame_roles:
                    val_key = f"{ev_id}_{'; '.join(frame_roles)}"
                    if val_key not in seen_valencies:
                        seen_valencies.add(val_key)
                        status = getattr(node, "truth_status", "True")
                        conf = getattr(node, "confidence", 1.0)
                        v_line = f"- Frame: {ev_disp} [{', '.join(frame_roles)}] | Status: {status} (conf={conf:.2f})."
                        valency_lines.append(v_line)

                # 3. Epistemic & Pragmatic Status
                ev_source = getattr(node, "evidence_source", None)
                if ev_source and ev_source != "direct_observation":
                    epistemic_lines.append(f"- Epistemic: {ev_disp} | Source: {ev_source}.")

            else:
                # Entity node
                if lbl and lbl != "?X":
                    entity_names.append(lbl)

                # Collect direct structural relations (CALLS, INHERITS_FROM, LOCATED_IN, etc.)
                if hasattr(node, "edges") and isinstance(node.edges, dict):
                    for rel, targets in node.edges.items():
                        if rel in (
                            "LOCATED_IN", "COUNTRY", "CAPITAL", "BORN_IN",
                            "INHERITS_FROM", "CALLS", "IMPLEMENTS", "IMPORTS",
                            "SHARES_BORDER", "PART_OF", "MEMBER_OF", "HEADQUARTERS",
                        ):
                            for t_cid in targets:
                                t_node = subgraph.get_node(t_cid)
                                t_lbl = _clean_node_label(t_node) if t_node else t_cid[:6]
                                relational_lines.append(f"- Fact: {lbl} -> {rel} -> {t_lbl}.")

        # Assemble briefing sections
        sections: List[str] = ["=== LOGICAL BRIEFING BLOCK ==="]

        if causal_temporal_lines:
            sections.append("--- Verified Causal & Temporal Paths ---")
            sections.extend(causal_temporal_lines)

        if valency_lines:
            sections.append("--- Thematic Valency Frames ---")
            sections.extend(valency_lines)

        if epistemic_lines:
            sections.append("--- Epistemic & Pragmatic Status ---")
            sections.extend(epistemic_lines)

        if relational_lines:
            sections.append("--- Key Relational Facts ---")
            sections.extend(relational_lines)

        if not causal_temporal_lines and not valency_lines and not relational_lines:
            if entity_names:
                unique_ents = list(dict.fromkeys(entity_names))
                sections.append(f"- Active Entities: {', '.join(unique_ents)}.")
            else:
                return ""

        briefing_text = "\n".join(sections).strip()
        if not briefing_text.endswith("."):
            briefing_text += "."

        return _truncate_to_token_budget(briefing_text, max_tokens)

    def assemble_passage_stream(
        self,
        passages: List[ProjectedPassage],
        max_tokens: int = 700,
    ) -> str:
        """Constructs Stream 2: Top-K Raw Source Passages from projected passages.

        Preserves 100% lexical fidelity of original text passages.

        Args:
            passages: List of ProjectedPassage objects.
            max_tokens: Token budget for the raw passage stream.

        Returns:
            Formatted Raw Source Passages string.
        """
        if not passages:
            return ""

        blocks: List[str] = ["=== RETRIEVED SOURCE PASSAGES ==="]
        accumulated_words = 0
        max_words = max(1, int(max_tokens / 1.33))

        for p in passages:
            if not p.text.strip():
                continue

            header = f"[Passage {p.passage_id} | Doc: {p.doc_id} | Score: {p.score:.2f}]"
            content = p.text.strip()
            block_text = f"{header}\n{content}"

            block_words = len(block_text.split())

            if accumulated_words + block_words <= max_words:
                blocks.append(block_text)
                accumulated_words += block_words
            elif len(blocks) == 1:
                # First passage exceeds budget: include with clean truncation
                truncated_content = _truncate_to_token_budget(content, max(1, int(max_tokens / 1.33) - 10))
                blocks.append(f"{header}\n{truncated_content}")
                break
            else:
                # Stop adding further passages
                break

        if len(blocks) == 1:
            return ""

        passage_text = "\n\n".join(blocks).strip()
        return passage_text

    def assemble_dual_stream_context(
        self,
        subgraph: QuantaGraph,
        passage_store: Optional[PassageStore] = None,
        max_tokens: int = 1000,
        query_text: str = "",
        activations: Optional[Dict[str, float]] = None,
    ) -> DualStreamContext:
        """Assembles the complete dual-stream context block for host LLMs.

        Dynamically balances token budgets between Stream 1 (Logical Briefing)
        and Stream 2 (Raw Source Passages).

        Args:
            subgraph: Retrieved QuantaGraph ASG sub-graph.
            passage_store: Optional PassageStore for passage retrieval.
            max_tokens: Maximum total token budget across both streams.
            query_text: Optional query text for filtering.
            activations: Optional dict of node CID -> activation score.

        Returns:
            DualStreamContext instance containing both streams and combined context.
        """
        store = passage_store or self.passage_store

        # 1. Project semantic scores onto raw passages
        ranked_passages = self.projector.project(
            subgraph=subgraph,
            passage_store=store,
            activations=activations,
        )

        has_passages = any(bool(p.text.strip()) for p in ranked_passages)

        # 2. Allocate token budgets
        if has_passages:
            logical_budget = max(10, int(max_tokens * self.logical_ratio))
            passage_budget = max(10, int(max_tokens * self.passage_ratio))

            briefing = self.assemble_logical_briefing(subgraph, max_tokens=logical_budget)

            # Roll over unconsumed logical budget to passage stream
            briefing_words = len(briefing.split()) if briefing else 0
            briefing_tokens = int(briefing_words * 1.33)
            remaining_for_passages = max(10, max_tokens - briefing_tokens)

            selected_passages = self.projector.select_top_passages(ranked_passages, max_tokens=remaining_for_passages)
            passage_stream = self.assemble_passage_stream(selected_passages, max_tokens=remaining_for_passages)
        else:
            briefing = self.assemble_logical_briefing(subgraph, max_tokens=max_tokens)
            selected_passages = []
            passage_stream = ""

        # 3. Assemble combined context
        if briefing and passage_stream:
            full_context = f"{briefing}\n\n{passage_stream}"
        elif briefing:
            full_context = briefing
        elif passage_stream:
            full_context = passage_stream
        else:
            full_context = ""

        # Enforce strict token budget truncation on full_context
        full_context = _truncate_to_token_budget(full_context, max_tokens)
        est_tokens = int(len(full_context.split()) * 1.33) if full_context else 0

        return DualStreamContext(
            logical_briefing=briefing,
            passage_stream=passage_stream,
            full_context=full_context,
            passages=selected_passages,
            token_count_estimate=est_tokens,
        )

    def format_context(
        self,
        subgraph: QuantaGraph,
        passage_store: Optional[PassageStore] = None,
        max_tokens: int = 500,
        query_text: str = "",
        activations: Optional[Dict[str, float]] = None,
    ) -> str:
        """Convenience method returning the assembled context string directly."""
        ctx = self.assemble_dual_stream_context(
            subgraph=subgraph,
            passage_store=passage_store,
            max_tokens=max_tokens,
            query_text=query_text,
            activations=activations,
        )
        return ctx.full_context


__all__ = [
    "BipartiteProjector",
    "DualStreamContext",
    "DualStreamContextAssembler",
    "ProjectedPassage",
]
