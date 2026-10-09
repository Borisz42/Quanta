"""Long-Context Benchmark Builder for QUANTA (§0.2).

Constructs standardized long-context evaluation benchmarks spanning:
1. MuSiQue-long: Multi-hop reasoning over distractor passages with gold paragraph position tracking.
2. NIAH: Synthetic needle-in-a-haystack placed at configurable depths across the length grid.
3. BABILong: Multi-hop dynamic state tracking embedded in long background distractors.

Writes deterministic, seeded dev/test splits to:
data/benchmarks/long_context/{dev,test}.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add repo root and src to sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmarks.suite_loaders import BenchmarkSuiteLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_long_context_bench")

DIVERSE_DISTRACTOR_PARAGRAPHS: List[str] = [
    "The celestial dynamics of the outer Magellanic Cloud exhibit intricate gravitational tidal perturbations induced by interaction with the Milky Way halo. Deep spectroscopic surveys reveal anomalous stellar kinematics across the leading arm, indicating that ancient dwarf galaxy mergers deposited substantial metal-poor globular clusters in the galactic periphery.",
    "Cryogenic engineering in superconducting magnetic resonance systems mandates rigorous thermal boundary management. Beryllium primary shield segments maintained at 4.2 Kelvin effectively mitigate ambient radiative heat leaks, while closed-cycle pulse-tube cryocoolers sustain ultra-stable operating states without continuous liquid helium replenishment.",
    "In distributed consensus protocols, Byzantine fault tolerance requires two-thirds supermajority quorum agreement across all active validator nodes. State machine replication under high network partitions relies on monotonic view-change counters and cryptographic threshold signatures to prevent double-spending and state bifurcation.",
    "The architecture of modern transactional databases guarantees serializable isolation through multi-version concurrency control paired with two-phase locking. Write-ahead logging ensures atomicity by persisting deterministic redo-undo records before physical memory pages are synchronized to solid-state storage arrays.",
    "Polymer matrix crystallization under high hydrostatic pressure demonstrates marked variations in lamellar thickness and spherulite growth rates. Molecular dynamics simulations show that cross-linking density at the amorphous-crystalline interface governs macroscopic tensile modulus and fracture resistance.",
    "Axiomatic formal verification employs first-order resolution refutation and satisfiability modulo theories to guarantee software invariant correctness. Inductive loop assertions and Hoare triple precondition checks eliminate buffer overrun vulnerabilities prior to compiler code generation.",
    "During the late Victorian maritime expedition of 1888, the survey vessel St. Jude documented uncharted archipelago bathymetry across the southern meridian. The naval logbooks recorded persistent prevailing westerlies, barometric fluctuations, and magnetic compass deviations near subterranean basaltic ridges.",
    "Autonomous swarm agents coordinate decentralized exploration via non-blocking gossip dissemination. Each localized peer maintains a compact Merkle state summary, exchanging peer digest vectors across gossip gossip rounds to achieve eventual consistency across unstable mesh topology.",
    "The thermodynamic efficiency of combined-cycle gas turbine systems approaches sixty-two percent under advanced aerodynamic turbine blade cooling. Ceramic thermal barrier coatings shield single-crystal superalloy substrates against combustion chamber gas temperatures exceeding sixteen hundred degrees Celsius.",
    "Cellular signal transduction through receptor tyrosine kinases regulates transcriptional activation and mitotic progression. Ligand-induced dimerization triggers intracellular autophosphorylation, initiating downstream kinase cascades that orchestrate chromatin remodeling and gene expression programs.",
    "Atmospheric composition measurements from airborne lidar systems identify microphysical aerosol interactions within the upper troposphere. Backscatter depolarization ratios indicate that heterogeneous ice nucleation is enhanced by mineral dust plumes transported across maritime convergence zones.",
    "The evolution of early medieval vernacular dialects reveals systematic phonological shifts governed by stress accentuation. Quantitative dialectometry confirms that lexical diffusion along riverine trade corridors accelerated the reduction of unstressed terminal inflectional suffixes.",
]

NIAH_NEEDLES: List[Tuple[str, str, str]] = [
    (
        "The designated mission access code for Vault 81 is PHANTOM-9092.",
        "What is the designated mission access code for Vault 81?",
        "PHANTOM-9092",
    ),
    (
        "The experimental superconducting quantum core operates at exact frequency 4.8872 GHz.",
        "What frequency does the experimental superconducting quantum core operate at?",
        "4.8872 GHz",
    ),
    (
        "Project Orion was designated under cryptographic reference DELTA-X99.",
        "What was Project Orion designated under?",
        "DELTA-X99",
    ),
    (
        "The emergency orbital override cipher for satellite constellation Aegis is CYGNUS-404.",
        "What is the emergency orbital override cipher for satellite constellation Aegis?",
        "CYGNUS-404",
    ),
    (
        "The primary chemical stabilizing agent synthesized at facility Gamma-9 is HEXAPHENYL-7.",
        "What is the primary chemical stabilizing agent synthesized at facility Gamma-9?",
        "HEXAPHENYL-7",
    ),
]

BABILONG_SCENARIOS: List[Dict[str, Any]] = [
    {
        "facts": [
            "Sandra journeyed to the garden.",
            "Daniel travelled to the kitchen.",
            "Sandra picked up the apple in the garden.",
            "Sandra journeyed to the bedroom.",
            "Sandra dropped the apple in the bedroom.",
            "Sandra walked to the office.",
        ],
        "question": "Where is the apple?",
        "answer": "bedroom",
    },
    {
        "facts": [
            "John travelled to the hallway.",
            "Mary moved to the office.",
            "John picked up the key in the hallway.",
            "John gave the key to Mary in the office.",
            "Mary travelled to the cellar.",
            "Mary put down the key in the cellar.",
            "Mary travelled to the garden.",
        ],
        "question": "Where is the key?",
        "answer": "cellar",
    },
    {
        "facts": [
            "Arthur placed the green container in the workshop.",
            "Arthur put the silver key inside the green container.",
            "Arthur carried the green container to the library.",
            "Arthur removed the silver key from the green container.",
            "Arthur left the silver key on the desk in the library.",
            "Arthur walked to the terrace.",
        ],
        "question": "Where is the silver key?",
        "answer": "library",
    },
    {
        "facts": [
            "The laboratory is north of the command center.",
            "The observatory is east of the laboratory.",
            "The hangar is south of the observatory.",
            "Dr. Chen walked from the command center into the laboratory.",
            "Dr. Chen then proceeded eastward into the observatory.",
        ],
        "question": "Which room is east of the laboratory?",
        "answer": "observatory",
    },
    {
        "facts": [
            "Bill travelled to the pantry.",
            "Bill acquired the lantern in the pantry.",
            "Bill journeyed to the attic.",
            "Bill dropped the lantern in the attic.",
            "Bill went back to the garden.",
        ],
        "question": "Where is the lantern?",
        "answer": "attic",
    },
]


@dataclass
class LongContextItem:
    """Standardized benchmark record."""
    id: str
    task_family: str  # musique, niah, babilong
    target_tokens: int
    prompt: str
    context: str
    gold_answer: str
    expected_tokens: List[str]
    gold_positions: List[int]  # 0-based paragraph index where gold evidence is placed
    gold_passages: List[str]
    needle_depth: Optional[float] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_family": self.task_family,
            "target_tokens": self.target_tokens,
            "prompt": self.prompt,
            "context": self.context,
            "gold_answer": self.gold_answer,
            "expected_tokens": self.expected_tokens,
            "gold_positions": self.gold_positions,
            "gold_passages": self.gold_passages,
            "needle_depth": self.needle_depth,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LongContextItem:
        return cls(
            id=data["id"],
            task_family=data["task_family"],
            target_tokens=data["target_tokens"],
            prompt=data["prompt"],
            context=data["context"],
            gold_answer=data["gold_answer"],
            expected_tokens=data.get("expected_tokens", [data["gold_answer"].lower()]),
            gold_positions=data.get("gold_positions", []),
            gold_passages=data.get("gold_passages", []),
            needle_depth=data.get("needle_depth"),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )


def estimate_tokens(text: str) -> int:
    """Standard token estimation heuristic (~1.33 tokens per whitespace word)."""
    words = len(text.strip().split())
    return max(1, int(words * 1.33))


class LongContextBenchBuilder:
    """Builds multi-hop and long-context datasets with accurate gold-position bookkeeping."""

    def __init__(
        self,
        musique_path: Optional[Path] = None,
        seed: int = 42,
    ):
        self.musique_path = musique_path or (repo_root / "data" / "benchmarks" / "musique_sample_real.json")
        self.seed = seed
        self.rng = random.Random(seed)
        self.musique_items = self._load_musique_items()

    def _load_musique_items(self) -> List[Dict[str, Any]]:
        """Loads source MuSiQue items and extracts a global pool of distractors."""
        if not self.musique_path.exists():
            logger.warning("MuSiQue sample file not found at %s. Using synthetic fallback.", self.musique_path)
            return []

        try:
            with open(self.musique_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            logger.info("Loaded %d MuSiQue items from %s", len(items), self.musique_path)
            return items
        except Exception as e:
            logger.error("Error reading %s: %s", self.musique_path, e)
            return []

    def _get_distractor_pool(self) -> List[str]:
        """Collects all distractor passages across MuSiQue items and diverse corpora."""
        pool: List[str] = list(DIVERSE_DISTRACTOR_PARAGRAPHS)
        for item in self.musique_items:
            for p in item.get("distractor_passages", []):
                clean_p = p.strip()
                if clean_p and clean_p not in pool:
                    pool.append(clean_p)
        return pool

    def build_musique_long(
        self,
        target_tokens: int,
        count: int = 2,
    ) -> List[LongContextItem]:
        """Pads MuSiQue items with distractors to reach target_tokens, tracking gold positions."""
        distractor_pool = self._get_distractor_pool()
        items: List[LongContextItem] = []

        # Target words heuristic
        target_words = int(target_tokens / 1.33)

        source_items = list(self.musique_items)
        if not source_items:
            # Fallback mock item
            source_items = [{
                "id": "musique_fallback_01",
                "question": "In which country was the director of the film born?",
                "gold_passages": [
                    "The film was directed by Paul Newman.",
                    "Paul Newman was born in the United States.",
                ],
                "answer": "United States",
            }]

        # Deterministic shuffle of items with local RNG
        local_rng = random.Random(self.seed + target_tokens + 101)
        shuffled_sources = list(source_items)
        local_rng.shuffle(shuffled_sources)

        for idx in range(count):
            base_item = shuffled_sources[idx % len(shuffled_sources)]
            golds = base_item.get("gold_passages", [])
            item_distractors = list(base_item.get("distractor_passages", []))

            # Calculate words needed from distractors
            gold_words = sum(len(g.split()) for g in golds)
            needed_distractor_words = max(0, target_words - gold_words)

            selected_distractors: List[str] = []
            cur_words = 0
            pool_idx = local_rng.randint(0, max(0, len(distractor_pool) - 1))

            while cur_words < needed_distractor_words:
                p = distractor_pool[pool_idx % len(distractor_pool)]
                selected_distractors.append(p)
                cur_words += len(p.split())
                pool_idx += 1

            # Interleave gold paragraphs across the distractors deterministically
            total_paras = len(selected_distractors) + len(golds)
            num_golds = len(golds)

            # Distribute gold positions evenly across context (e.g. 20%, 50%, 80%)
            gold_positions: List[int] = []
            for g_idx in range(num_golds):
                fraction = (g_idx + 1) / float(num_golds + 1)
                pos = int(total_paras * fraction)
                # Ensure no position collision
                while pos in gold_positions:
                    pos = (pos + 1) % total_paras
                gold_positions.append(pos)
            gold_positions.sort()

            # Assemble full paragraphs list
            assembled_paragraphs: List[str] = []
            g_ptr = 0
            d_ptr = 0
            for p_i in range(total_paras):
                if p_i in gold_positions:
                    assembled_paragraphs.append(golds[g_ptr])
                    g_ptr += 1
                else:
                    if d_ptr < len(selected_distractors):
                        assembled_paragraphs.append(selected_distractors[d_ptr])
                        d_ptr += 1
                    else:
                        assembled_paragraphs.append(distractor_pool[(pool_idx + p_i) % len(distractor_pool)])

            # Verify actual gold positions match
            actual_positions: List[int] = []
            for p_i, p_text in enumerate(assembled_paragraphs):
                if any(g in p_text for g in golds):
                    actual_positions.append(p_i)

            context_str = "\n\n".join(
                f"Paragraph [{i+1}]: {p}" for i, p in enumerate(assembled_paragraphs)
            )
            q = base_item.get("question", "")
            ans = str(base_item.get("answer", "")).strip()

            full_prompt = (
                f"{q}\n\n"
                "Instructions:\n"
                "1. Answer the question directly and concisely based strictly on the provided paragraphs.\n"
                "2. Conclude on a new line in the exact format: 'Answer: <final answer>'."
            )
            token_est = estimate_tokens(context_str + "\n" + full_prompt)

            items.append(
                LongContextItem(
                    id=f"musique_{target_tokens // 1000}k_{idx+1}",
                    task_family="musique",
                    target_tokens=target_tokens,
                    prompt=full_prompt,
                    context=context_str,
                    gold_answer=ans,
                    expected_tokens=[ans.lower()],
                    gold_positions=actual_positions,
                    gold_passages=golds,
                    needle_depth=round(actual_positions[0] / max(1, len(assembled_paragraphs)), 3) if actual_positions else None,
                    token_count=token_est,
                    metadata={
                        "original_id": base_item.get("id"),
                        "hop_count": base_item.get("hop_count", 2),
                        "total_paragraphs": len(assembled_paragraphs),
                    },
                )
            )

        return items

    def build_niah(
        self,
        target_tokens: int,
        count: int = 2,
    ) -> List[LongContextItem]:
        """Places synthetic needle at configurable depths across the target length grid."""
        items: List[LongContextItem] = []
        target_words = int(target_tokens / 1.33)
        distractor_pool = self._get_distractor_pool()

        local_rng = random.Random(self.seed + target_tokens + 202)
        # Depth variants across the document (10%, 25%, 50%, 75%, 90%)
        depth_grid = [0.10, 0.25, 0.50, 0.75, 0.90]

        for idx in range(count):
            needle_tuple = NIAH_NEEDLES[idx % len(NIAH_NEEDLES)]
            needle_text, q, ans = needle_tuple
            depth = depth_grid[idx % len(depth_grid)]

            # Assemble background distractors
            paragraphs: List[str] = []
            cur_words = 0
            pool_idx = local_rng.randint(0, len(distractor_pool) - 1)
            while cur_words < target_words:
                p = distractor_pool[pool_idx % len(distractor_pool)]
                paragraphs.append(p)
                cur_words += len(p.split())
                pool_idx += 1

            # Insert needle at specified depth
            insert_pos = max(0, min(len(paragraphs), int(len(paragraphs) * depth)))
            needle_para = f"CRITICAL LOG ENTRY: {needle_text}"
            paragraphs.insert(insert_pos, needle_para)

            context_str = "\n\n".join(
                f"Section [{i+1}]: {p}" for i, p in enumerate(paragraphs)
            )
            full_prompt = (
                f"{q}\n\n"
                "Instructions: Answer directly using the exact code or value from the record. "
                "Conclude with: 'Answer: <final answer>'."
            )
            token_est = estimate_tokens(context_str + "\n" + full_prompt)

            items.append(
                LongContextItem(
                    id=f"niah_{target_tokens // 1000}k_d{int(depth*100)}_{idx+1}",
                    task_family="niah",
                    target_tokens=target_tokens,
                    prompt=full_prompt,
                    context=context_str,
                    gold_answer=ans,
                    expected_tokens=[ans.lower()],
                    gold_positions=[insert_pos],
                    gold_passages=[needle_text],
                    needle_depth=depth,
                    token_count=token_est,
                    metadata={
                        "needle_text": needle_text,
                        "total_sections": len(paragraphs),
                    },
                )
            )

        return items

    def build_babilong(
        self,
        target_tokens: int,
        count: int = 2,
    ) -> List[LongContextItem]:
        """Embeds multi-hop state tracking facts at distributed depths across background text."""
        items: List[LongContextItem] = []
        target_words = int(target_tokens / 1.33)
        distractor_pool = self._get_distractor_pool()

        local_rng = random.Random(self.seed + target_tokens + 303)

        for idx in range(count):
            scenario = BABILONG_SCENARIOS[idx % len(BABILONG_SCENARIOS)]
            facts = scenario["facts"]
            q = scenario["question"]
            ans = scenario["answer"]

            # Assemble background paragraphs
            paragraphs: List[str] = []
            cur_words = 0
            pool_idx = local_rng.randint(0, len(distractor_pool) - 1)
            while cur_words < target_words:
                p = distractor_pool[pool_idx % len(distractor_pool)]
                paragraphs.append(p)
                cur_words += len(p.split())
                pool_idx += 1

            # Insert facts across distributed points
            gold_positions: List[int] = []
            num_facts = len(facts)
            for f_i, fact in enumerate(facts):
                fraction = (f_i + 1) / float(num_facts + 1)
                insert_idx = int(len(paragraphs) * fraction)
                paragraphs.insert(insert_idx, f"State Update: {fact}")
                gold_positions.append(insert_idx)

            # Re-verify exact indices of facts
            actual_positions: List[int] = []
            for p_i, p in enumerate(paragraphs):
                if any(fact in p for fact in facts):
                    actual_positions.append(p_i)

            context_str = "\n\n".join(
                f"Narrative Block [{i+1}]: {p}" for i, p in enumerate(paragraphs)
            )
            full_prompt = (
                f"{q}\n\n"
                "Instructions: Track the dynamic locations and entity movements in the narrative. "
                "Conclude with: 'Answer: <final answer>'."
            )
            token_est = estimate_tokens(context_str + "\n" + full_prompt)

            items.append(
                LongContextItem(
                    id=f"babilong_{target_tokens // 1000}k_{idx+1}",
                    task_family="babilong",
                    target_tokens=target_tokens,
                    prompt=full_prompt,
                    context=context_str,
                    gold_answer=ans,
                    expected_tokens=[ans.lower()],
                    gold_positions=actual_positions,
                    gold_passages=facts,
                    needle_depth=round(actual_positions[0] / max(1, len(paragraphs)), 3) if actual_positions else None,
                    token_count=token_est,
                    metadata={
                        "facts_count": len(facts),
                        "total_blocks": len(paragraphs),
                    },
                )
            )

        return items

    def build_dataset(
        self,
        length_grid: Sequence[int],
        samples_per_cell: int = 2,
    ) -> List[LongContextItem]:
        """Generates all items across families and length grid."""
        all_items: List[LongContextItem] = []
        for length in length_grid:
            logger.info("Generating length %d tokens...", length)
            m_items = self.build_musique_long(length, count=samples_per_cell)
            n_items = self.build_niah(length, count=samples_per_cell)
            b_items = self.build_babilong(length, count=samples_per_cell)

            all_items.extend(m_items)
            all_items.extend(n_items)
            all_items.extend(b_items)

        logger.info("Generated %d total items across %d length buckets", len(all_items), len(length_grid))
        return all_items

    def write_splits(
        self,
        items: List[LongContextItem],
        output_dir: Path,
        dev_ratio: float = 0.5,
    ) -> Tuple[Path, Path]:
        """Deterministically partitions generated items into dev and test splits."""
        output_dir.mkdir(parents=True, exist_ok=True)
        dev_file = output_dir / "dev.jsonl"
        test_file = output_dir / "test.jsonl"

        dev_items: List[LongContextItem] = []
        test_items: List[LongContextItem] = []

        # Stratify by (task_family, target_tokens) to ensure balanced splits
        buckets: Dict[Tuple[str, int], List[LongContextItem]] = {}
        for item in items:
            key = (item.task_family, item.target_tokens)
            buckets.setdefault(key, []).append(item)

        split_rng = random.Random(self.seed + 999)
        for key, bucket_items in sorted(buckets.items()):
            # Deterministic alternate assignment for perfectly balanced dev/test
            for i, itm in enumerate(bucket_items):
                if i % 2 == 0:
                    dev_items.append(itm)
                else:
                    test_items.append(itm)

        # Write dev
        with open(dev_file, "w", encoding="utf-8") as f:
            for itm in dev_items:
                f.write(json.dumps(itm.to_dict()) + "\n")

        # Write test
        with open(test_file, "w", encoding="utf-8") as f:
            for itm in test_items:
                f.write(json.dumps(itm.to_dict()) + "\n")

        manifest = {
            "seed": self.seed,
            "total_samples": len(items),
            "dev_samples": len(dev_items),
            "test_samples": len(test_items),
            "dev_file": str(dev_file),
            "test_file": str(test_file),
        }
        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Wrote %d dev items to %s and %d test items to %s",
            len(dev_items), dev_file, len(test_items), test_file
        )
        return dev_file, test_file


def main():
    parser = argparse.ArgumentParser(description="QUANTA Long-Context Benchmark Builder")
    parser.add_argument("--pilot", action="store_true", help="Build small pilot grid (2k, 4k tokens, 2 samples/cell)")
    parser.add_argument("--length-grid", type=str, default=None, help="Comma-separated tokens grid (e.g. 2000,4000,8000)")
    parser.add_argument("--samples-per-cell", type=int, default=None, help="Number of samples per length cell per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    parser.add_argument("--output-dir", type=str, default="data/benchmarks/long_context", help="Output directory")
    args = parser.parse_args()

    if args.pilot:
        length_grid = [2000, 4000]
        samples_per_cell = 2
    else:
        if args.length_grid:
            length_grid = [int(x.strip()) for x in args.length_grid.split(",") if x.strip()]
        else:
            length_grid = [2000, 4000]
        samples_per_cell = args.samples_per_cell or 2

    builder = LongContextBenchBuilder(seed=args.seed)
    items = builder.build_dataset(length_grid=length_grid, samples_per_cell=samples_per_cell)
    output_dir = Path(args.output_dir)
    builder.write_splits(items=items, output_dir=output_dir)


if __name__ == "__main__":
    main()
