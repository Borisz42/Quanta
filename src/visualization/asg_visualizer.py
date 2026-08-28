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
from core.slots import get_slot_by_index
from core.types import QuaternaryValue


class ASGVisualizer:
    """Renders QuantaGraph and QuantaNodes into structured ASCII/Unicode art, tables, and Mermaid graphs."""

    @staticmethod
    def _val_name(val: int | QuaternaryValue) -> str:
        v = int(val)
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
                slot_strs = [f"{get_slot_by_index(k).name}: {cls._val_name(v)}" for k, v in node.vector.active_slots().items()]
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

    @classmethod
    def render_node_details_table(cls, graph: QuantaGraph, style: str = "unicode") -> str:
        """Renders a structured ASCII table describing each node and its slot assignments."""
        if not graph or not graph.nodes:
            return "No nodes in graph."

        string_io = io.StringIO()
        console = Console(file=string_io, width=120, color_system=None)
        box_style = ASCII if style == "ascii" else ROUNDED

        table = Table(title="QUANTA ASG Node Inspection", box=box_style, show_header=True, header_style="bold")
        table.add_column("Node CID", style="bold", width=12)
        table.add_column("Anchor / Literal", width=22)
        table.add_column("Outgoing Edges", width=34)
        table.add_column("Active Semantic Slots (Top / All)", width=44)

        for cid in sorted(graph.nodes.keys()):
            node = graph.nodes[cid]
            is_root = " (ROOT)" if cid == graph.root_cid else ""
            short_cid = f"{cid[:8]}...{is_root}"
            
            anchor_text = f"Anchor: '{node.anchor}'" if node.anchor else "(None)"
            if node.literal is not None:
                anchor_text += f"\nLiteral: {node.literal}"

            edges_lines = []
            for rel in sorted(node.edges.keys()):
                for tgt in sorted(node.edges[rel]):
                    tgt_node = graph.nodes.get(tgt)
                    tgt_name = f"'{tgt_node.anchor}'" if tgt_node and tgt_node.anchor else tgt[:8]
                    edges_lines.append(f"• {rel} ──> [{tgt[:8]}] {tgt_name}")
            edges_text = "\n".join(edges_lines) if edges_lines else "(No outgoing edges)"

            slots_lines = []
            active = node.vector.active_slots()
            for k, v in active.items():
                slot_meta = get_slot_by_index(k)
                slots_lines.append(f"• {slot_meta.name}: {cls._val_name(v)}")
            slots_text = "\n".join(slots_lines) if slots_lines else "(Empty / All Irrelevant)"

            table.add_row(
                escape(short_cid),
                escape(anchor_text),
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
