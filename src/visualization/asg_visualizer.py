"""ASG Graph and Quanta Node Visualizer using rich and ASCII/Unicode art."""

from __future__ import annotations
import io
import json
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.box import ROUNDED, ASCII
from rich.markup import escape

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_index, SlotBand, SlotDefinition, get_slot_contract
from core.types import QuaternaryValue, BandContract


class ASGVisualizer:
    """Renders QuantaGraph and QuantaNodes into structured ASCII/Unicode art, tables, and Mermaid graphs."""

    @staticmethod
    def _val_name(val: int | QuaternaryValue, slot_def: Optional[SlotDefinition] = None) -> str:
        v = int(val)
        if slot_def is not None:
            contract = slot_def.contract
            if contract == BandContract.STRUCTURAL:
                if v == 1:
                    return "ACTIVE_LOCAL (1)"
                elif v == 2:
                    return "ACTIVE_EXTERNAL (2)"
                elif v == 3:
                    return "ACTIVE_MERKLE (3)"
                return "INACTIVE (0)"
            elif contract == BandContract.REGISTER:
                if v == 1:
                    return "BOUND_LOCAL (1)"
                elif v == 2:
                    return "BOUND_EXTERNAL (2)"
                elif v == 3:
                    return "QUERY_TARGET (3)"
                return "UNBOUND (0)"

        if v == 1:
            return "TRUE (1)"
        elif v == 2:
            return "FALSE (2)"
        elif v == 3:
            return "UNKNOWN (3)"
        return "IRRELEVANT (0)"

    @classmethod
    def render_ascii_tree(cls, graph: QuantaGraph, style: str = "unicode", show_slots: bool = False) -> str:
        """Renders the graph hierarchy as an ASCII or Unicode tree."""
        if not graph or not graph.nodes:
            return "[Empty Graph]"

        string_io = io.StringIO()
        console = Console(file=string_io, width=120, color_system=None)

        nodes = graph.nodes
        visited: Set[str] = set()

        def _format_node_label(cid: str) -> str:
            n = nodes.get(cid)
            if not n:
                return f"[Missing Node {cid[:8]}...]"
            anchor = f"'{n.anchor}'" if n.anchor else "(no-anchor)"
            literal = f" = {json.dumps(n.literal)}" if n.literal is not None else ""
            slot_count = len(n.vector.active_slots())
            return f"Node [{cid[:8]}] Concept: {anchor}{literal} ({slot_count} active slots)"

        def _build_tree(parent_tree: Tree, current_cid: str, path: Set[str]):
            if current_cid in path:
                parent_tree.add(escape(f"[Cycle / Back-reference -> {current_cid[:8]}]"))
                return

            visited.add(current_cid)
            node = nodes.get(current_cid)
            if not node:
                return

            if show_slots and node.vector.active_slots():
                slot_strs = [f"{get_slot_by_index(k).name}: {cls._val_name(v, slot_def=get_slot_by_index(k))}" for k, v in node.vector.active_slots().items()]
                parent_tree.add(escape(f"Slots: [{', '.join(slot_strs[:6])}{'...' if len(slot_strs) > 6 else ''}]"))

            for rel in sorted(node.edges.keys()):
                targets = sorted(node.edges[rel])
                for target_cid in targets:
                    target_label = _format_node_label(target_cid)
                    edge_prefix = f"({rel}) ──> "
                    child_branch = parent_tree.add(escape(f"{edge_prefix}{target_label}"))
                    _build_tree(child_branch, target_cid, path | {current_cid})

        # Start from root node if available
        root_cid = graph.root_cid if (graph.root_cid and graph.root_cid in nodes) else next(iter(nodes.keys()))
        root_tree = Tree(escape(_format_node_label(root_cid)))
        _build_tree(root_tree, root_cid, set())

        console.print(root_tree)

        # Handle any disconnected components
        unvisited = set(nodes.keys()) - visited
        if unvisited:
            for orphan_cid in sorted(unvisited):
                orphan_tree = Tree(escape(f"[Unlinked Component] {_format_node_label(orphan_cid)}"))
                _build_tree(orphan_tree, orphan_cid, set())
                console.print(orphan_tree)

        return string_io.getvalue().rstrip()

    BAND_TITLES: Dict[SlotBand, str] = {
        SlotBand.BAND_0_NSM_KINEMATICS: "Band 0 (NSM Primes & Kinematics)",
        SlotBand.BAND_1_VALENCIES_TOPOLOGY: "Band 1 (Valencies & Topology)",
        SlotBand.BAND_2_LOGIC_VARIABLES: "Band 2 (Logic & Variables)",
        SlotBand.BAND_3_ONTOLOGY_STRUCTURES: "Band 3 (Ontology & Structures)",
        SlotBand.BAND_4_AFFORDANCES_OPERATIONS: "Band 4 (Affordances & Operations)",
        SlotBand.BAND_5_TOM_PRAGMATICS: "Band 5 (Theory of Mind & Pragmatics)",
        SlotBand.BAND_6_PROOF_DEONTICS: "Band 6 (Proof Solvers & Deontics)",
        SlotBand.BAND_7_SPATIOTEMPORAL_CAUSAL: "Band 7 (Spatio-Temporal & Causal)",
    }

    @classmethod
    def render_node_details_table(cls, graph: QuantaGraph, style: str = "unicode") -> str:
        """Renders a structured ASCII/Unicode table describing each node and its slot assignments grouped by band."""
        if not graph or not graph.nodes:
            return "No nodes in graph."

        string_io = io.StringIO()
        console = Console(file=string_io, width=120, color_system=None)
        box_style = ASCII if style == "ascii" else ROUNDED
        arrow = "──>" if style != "ascii" else "-->"
        bullet = "•" if style != "ascii" else "*"

        table = Table(
            title="QUANTA ASG Node Specifications & Semantic Slots",
            box=box_style,
            show_header=True,
            header_style="bold",
            show_lines=True,
        )
        table.add_column("Concept / Anchor", width=28, style="bold")
        table.add_column("Outgoing Edges / Valencies", width=38)
        table.add_column("Active Semantic Slots (Grouped by Band)", width=50)

        # Order nodes: ROOT first, then BFS traversal of reachable nodes, then unvisited
        ordered_cids: List[str] = []
        visited: Set[str] = set()
        root_node = graph.root
        root_cid = root_node.cid if root_node else (graph.root_cid if graph.root_cid in graph.nodes else (next(iter(graph.nodes.keys())) if graph.nodes else None))

        if root_cid and root_cid in graph.nodes:
            queue = [root_cid]
            while queue:
                curr = queue.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    ordered_cids.append(curr)
                    node = graph.nodes.get(curr)
                    if node:
                        for rel in sorted(node.edges.keys()):
                            for tgt in sorted(node.edges[rel]):
                                if tgt not in visited and tgt in graph.nodes:
                                    queue.append(tgt)
        for cid in sorted(graph.nodes.keys()):
            if cid not in visited:
                ordered_cids.append(cid)

        for cid in ordered_cids:
            node = graph.nodes[cid]
            is_root = " [ROOT]" if (node is root_node or cid == root_cid) else ""

            anchor_lines = []
            if node.anchor:
                anchor_lines.append(f"Anchor: '{node.anchor}'{is_root}")
            else:
                anchor_lines.append(f"(No Anchor){is_root}")
            if node.literal is not None:
                lit_str = f'"{node.literal}"' if isinstance(node.literal, str) else f"{node.literal}"
                anchor_lines.append(f"Literal: {lit_str}")
            active_slots = node.vector.active_slots()
            anchor_lines.append(f"Active Slots: {len(active_slots)}")
            concept_text = "\n".join(anchor_lines)

            # Format Edges
            edges_lines = []
            for rel in sorted(node.edges.keys()):
                for tgt in sorted(node.edges[rel]):
                    tgt_node = graph.nodes.get(tgt)
                    if tgt_node:
                        if tgt_node.anchor:
                            tgt_desc = f"'{tgt_node.anchor}'"
                            if tgt_node.literal:
                                tgt_desc += f' ("{tgt_node.literal}")'
                        elif tgt_node.literal:
                            tgt_desc = f'"{tgt_node.literal}"'
                        else:
                            tgt_desc = f"Node [{tgt[:8]}]"
                    else:
                        tgt_desc = f"Node [{tgt[:8]}]"
                    edges_lines.append(f"{bullet} {rel} {arrow} {tgt_desc}")
            edges_text = "\n".join(edges_lines) if edges_lines else "(No outgoing edges / Leaf)"

            # Group active slots by Band (0-3)
            slots_by_band: Dict[SlotBand, List[Any]] = {}
            for k, v in active_slots.items():
                slot_meta = get_slot_by_index(k)
                band = slot_meta.band
                if band not in slots_by_band:
                    slots_by_band[band] = []
                slots_by_band[band].append((slot_meta, v))

            slots_lines = []
            if not slots_by_band:
                slots_text = "(Empty / All Irrelevant)"
            else:
                for band in sorted(slots_by_band.keys()):
                    band_title = cls.BAND_TITLES.get(band, f"Band {int(band)}")
                    slots_lines.append(f"[{band_title}]")
                    for s_def, val in slots_by_band[band]:
                        slots_lines.append(f"  {bullet} {s_def.name}: {cls._val_name(val)}")
                slots_text = "\n".join(slots_lines)

            table.add_row(
                escape(concept_text),
                escape(edges_text),
                escape(slots_text),
            )

        console.print(table)
        return string_io.getvalue().rstrip()

    @classmethod
    def render_mermaid(cls, graph: QuantaGraph) -> str:
        """Renders the graph into Mermaid flowchart syntax for Markdown documents."""
        if not graph or not graph.nodes:
            return "```mermaid\nflowchart TD\n    empty[Empty Graph]\n```"

        lines = ["```mermaid", "flowchart TD"]
        
        # Node declarations
        for cid, node in sorted(graph.nodes.items()):
            node_id = f"node_{cid[:8]}"
            anchor = (node.anchor or "node").replace('"', "'")
            is_root = " ⭐[ROOT]" if cid == graph.root_cid else ""
            literal_val = json.dumps(node.literal).replace('"', "'") if node.literal is not None else None
            literal = f"<br/>lit: {literal_val}" if literal_val is not None else ""
            slot_count = len(node.vector.active_slots())
            
            label = f'"{anchor}{is_root}<br/><small>CID: {cid[:8]} | {slot_count} slots</small>{literal}"'
            lines.append(f"    {node_id}[{label}]")

        # Edge declarations
        has_edges = False
        for src_cid, node in sorted(graph.nodes.items()):
            src_id = f"node_{src_cid[:8]}"
            for rel in sorted(node.edges.keys()):
                for tgt_cid in sorted(node.edges[rel]):
                    tgt_id = f"node_{tgt_cid[:8]}"
                    tgt_node = graph.get_node(tgt_cid)
                    # Style edge based on Merkle fold vs normal
                    if tgt_node and tgt_node.anchor and tgt_node.anchor.startswith("merkle:"):
                        lines.append(f"    {src_id} ==>|{rel} (Merkle CID)| {tgt_id}")
                    elif rel in ("GRAPH_EXT_REFERENCE", "GRAPH_SQLITE_REF"):
                        lines.append(f"    {src_id} -.->|{rel} (External)| {tgt_id}")
                    else:
                        lines.append(f"    {src_id} -->|{rel}| {tgt_id}")
                    has_edges = True

        if not has_edges and len(graph.nodes) > 1:
            lines.append("    %% Note: Disconnected components")

        lines.append("```")
        return "\n".join(lines)

    @classmethod
    def format_translation_block(
        cls,
        example_id: int,
        source_text: str,
        target_text: str,
        source_modality: str,
        target_modality: str,
        graph: QuantaGraph,
        merkle_root: Optional[str] = None,
        is_valid: bool = True,
        preservation_rate: Optional[float] = None,
        as_markdown: bool = True,
    ) -> str:
        """Formats a full translation example with ASCII tree, node table, and optional Mermaid chart."""
        status_str = "VALID (PASS)" if is_valid else "INVALID (FAIL)"
        merkle_str = merkle_root or (graph.compute_merkle_root() if graph else "N/A")
        
        tree_art = cls.render_ascii_tree(graph, style="unicode" if as_markdown else "ascii")
        node_table = cls.render_node_details_table(graph, style="unicode" if as_markdown else "ascii")

        if as_markdown:
            mermaid_chart = cls.render_mermaid(graph)
            pres_str = f"- **Slot Preservation Rate:** `{preservation_rate:.1%}`  \n" if preservation_rate is not None else ""
            md_output = f"""### Example {example_id:02d}: {source_modality.upper()} → {target_modality.upper()}

- **Input ({source_modality}):** `{source_text}`
- **Output ({target_modality}):** `{target_text}`
- **Validation Gate:** `{status_str}`
- **BLAKE3 Merkle Root:** `{merkle_str}`
{pres_str}
#### ASG Graph Hierarchy (ASCII Art)
```text
{tree_art}
```

#### Interactive ASG Diagram (Mermaid)
{mermaid_chart}

#### Node Specifications & Semantic Slots
```text
{node_table}
```

---
"""
            return md_output
        else:
            pres_str = f"Preservation Rate: {preservation_rate:.1%}\n" if preservation_rate is not None else ""
            txt_output = f"""================================================================================
EXAMPLE {example_id:02d} [{source_modality.upper()} -> {target_modality.upper()}]
================================================================================
Input:       {source_text}
Output:      {target_text}
Validation:  {status_str}
Merkle Root: {merkle_str}
{pres_str}
--- ASG GRAPH HIERARCHY ---
{tree_art}

--- NODE SPECIFICATIONS & SEMANTIC SLOTS ---
{node_table}

"""
            return txt_output
