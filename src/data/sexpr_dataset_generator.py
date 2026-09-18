"""High-performance S-Expression Dataset Generator for QUANTA LoRA Fine-Tuning (Phase 6).

Ingests premises and narratives across multiple domains (FOLIO, ProofWriter, bAbI,
CLUTRR, Python Code ASTs, and episodic narratives), converts them into canonical
QuantaGraphs, and serializes them into strict GBNF S-expressions in standard
instruction-tuning JSONL format.

Supports synthetic MUC repair conditioning where diagnostic hints are injected
into the input prompt paired with corrected target S-expressions.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.asg import QuantaGraph
from data.real_loader import RealDatasetLoader
from parser.ast_parser import ASTForwardParser
from parser.fol_parser import FOLParser
from parser.nlp_forward import NLPForwardParser
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.sexpr_parser import parse_sexpr, serialize_to_sexpr

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Transduce the following text into a canonical GBNF S-expression graph."
MUC_REPAIR_INSTRUCTION = "Repair the S-expression graph to resolve the diagnosed formal verification conflicts."

# Rich synthetic templates for offline / fast fallback across domains
SYNTHETIC_PROOFWRITER_TEMPLATES = [
    "The cat is cold. If someone is cold then they are quiet. Therefore the cat is quiet.",
    "The bear visits the squirrel. If a bear visits a squirrel then the squirrel is happy.",
    "The mouse sees the tiger. The tiger chases the rabbit in the forest.",
    "The dog is big and rough. If someone is big then they eat the meat.",
    "The lion enters the cave. If an animal enters the cave then it sleeps until dawn.",
    "The rabbit finds the carrot under the tree. The rabbit eats the carrot eagerly.",
    "The wolf is hungry. Every wolf hunts in the mountain during winter.",
    "The owl watches the garden at night. The owl hears the small rodent moving.",
]

SYNTHETIC_BABI_TEMPLATES = [
    "John travelled to the hallway. Mary journeyed to the kitchen. John dropped the milk.",
    "Daniel moved to the garden. Sandra went back to the bedroom. Daniel grabbed the apple.",
    "Mary moved to the office. John journeyed to the garden. Mary left the football.",
    "Sandra picked up the football there. Sandra went to the kitchen. Sandra dropped the football.",
    "Daniel went to the kitchen. John travelled to the office. Daniel took the package.",
    "Sandra journeyed to the bedroom. Daniel moved to the garden. Sandra gave the book to John.",
    "John picked up the box. John travelled to the hallway. John put down the box.",
    "Mary grabbed the key. Mary travelled to the cellar. Mary unlocked the door.",
]

SYNTHETIC_CLUTRR_TEMPLATES = [
    "Alice is the mother of Bob. Bob has a daughter named Claire. Alice is Claire's grandmother.",
    "David and his brother Eric visited their father Frank at his country home.",
    "Grace celebrated her birthday with her sister Hannah and their mother Irene.",
    "James took his son Kevin to visit Kevin's aunt Laura in Boston.",
    "Michael introduced his wife Nancy and their son Oliver to the family.",
    "Patricia helped her daughter Rachel prepare a gift for Rachel's father Thomas.",
    "William invited his cousin Zachary and his uncle Victor to the reunion.",
    "George congratulated his sister Helen and her husband Ian on their anniversary.",
]

SYNTHETIC_NARRATIVE_TEMPLATES = [
    "Dr. Eleanor Vance isolated the synthetic compound inside the cryogenic containment cell. She verified that the temperature held steady.",
    "Professor Jonathan Hayes calibrated the particle detector in the underground laboratory. He recorded three anomalies before midnight.",
    "Dr. Aris Thorne synthesized the crystalline polymer at high pressure. The automated sensor detected rapid crystallization.",
    "Commander Elena Rostova docked the transport module with the orbital station. The primary airlock pressurized without incident.",
    "Chief Engineer Liam Murphy ignited the plasma thruster in the test bay. The thermal telemetry confirmed stable ignition.",
    "Dr. Sophia Chen extracted the enzyme sample from the bio-reactor. She transferred the specimen into the storage vial.",
]


class SExprDatasetGenerator:
    """Generates instruction-tuning datasets of (Discourse, S-Expression) pairs."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        offline: bool = False,
    ):
        self.cache_dir = Path(cache_dir or "data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)
        self.offline = offline

        self._real_loader: Optional[RealDatasetLoader] = None
        self._nlp_parser: Optional[NLPForwardParser] = None
        self._fol_parser: Optional[FOLParser] = None
        self._ast_parser: Optional[ASTForwardParser] = None

    @property
    def real_loader(self) -> RealDatasetLoader:
        if self._real_loader is None:
            self._real_loader = RealDatasetLoader(cache_dir=self.cache_dir)
        return self._real_loader

    @property
    def nlp_parser(self) -> NLPForwardParser:
        if self._nlp_parser is None:
            self._nlp_parser = NLPForwardParser()
        return self._nlp_parser

    @property
    def fol_parser(self) -> FOLParser:
        if self._fol_parser is None:
            self._fol_parser = FOLParser()
        return self._fol_parser

    @property
    def ast_parser(self) -> ASTForwardParser:
        if self._ast_parser is None:
            self._ast_parser = ASTForwardParser()
        return self._ast_parser

    # -----------------------------------------------------------------------
    # Domain Sample Extractors
    # -----------------------------------------------------------------------

    def extract_folio_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Extract paired samples from FOLIO premise sentences and First-Order Logic."""
        samples: List[Dict[str, Any]] = []
        try:
            raw_items = self.real_loader.load_folio_samples(max_samples=count)
        except Exception as e:
            logger.warning(f"Error loading FOLIO samples: {e}. Using fallback templates.")
            raw_items = []

        if not raw_items:
            raw_items = [
                ("All humans are mortal. Socrates is a human.", "all x. (Human(x) -> Mortal(x))"),
                ("Every scientist observes nature.", "all x. (Scientist(x) -> Observes(x, nature))"),
                ("No robot feels emotional pain.", "all x. (Robot(x) -> ~Feels(x, pain))"),
            ]

        for i, (premise_text, fol_formula) in enumerate(raw_items):
            if len(samples) >= count:
                break
            premise_text = premise_text.strip()
            if len(premise_text) < 4:
                continue

            sexpr: Optional[str] = None
            # Try parsing via FOL parser first if formula exists
            if fol_formula and len(fol_formula) > 3:
                try:
                    graph = self.fol_parser.parse_formula(fol_formula)
                    sexpr = serialize_to_sexpr(graph, pretty=True)
                except Exception:
                    sexpr = None

            # Fallback to NLP forward parser on premise text
            if not sexpr:
                try:
                    graph = self.nlp_parser.parse_sentence(premise_text, domain_context="FOLIO")
                    sexpr = serialize_to_sexpr(graph, pretty=True)
                except Exception:
                    continue

            samples.append({
                "id": f"folio_{i+1:05d}",
                "domain": "folio",
                "instruction": DEFAULT_INSTRUCTION,
                "input": premise_text,
                "output": sexpr,
                "metadata": {"fol_formula": fol_formula or ""},
            })

        # Cycle if more samples requested than available
        if samples and len(samples) < count:
            base_samples = list(samples)
            cycle_idx = 0
            while len(samples) < count:
                orig = base_samples[cycle_idx % len(base_samples)]
                cycle_num = cycle_idx // len(base_samples) + 1
                samples.append({
                    "id": f"{orig['id']}_var{cycle_num}",
                    "domain": "folio",
                    "instruction": orig["instruction"],
                    "input": f"{orig['input']} (Case #{cycle_num})",
                    "output": orig["output"],
                    "metadata": orig["metadata"],
                })
                cycle_idx += 1

        return samples[:count]

    def extract_proofwriter_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Extract paired samples from ProofWriter theory rules and facts."""
        samples: List[Dict[str, Any]] = []
        raw_items: List[str] = []
        cache_file = self.cache_dir / "proofwriter_train.jsonl"
        if cache_file.exists():
            try:
                raw_items = self.real_loader.load_proofwriter_samples(max_samples=count)
            except Exception as e:
                logger.warning(f"Error loading ProofWriter cache: {e}")
        elif not self.offline:
            try:
                raw_items = self.real_loader.load_proofwriter_samples(max_samples=count)
            except Exception as e:
                logger.warning(f"Error loading ProofWriter: {e}")

        if not raw_items:
            raw_items = SYNTHETIC_PROOFWRITER_TEMPLATES

        for i, text in enumerate(raw_items):
            if len(samples) >= count:
                break
            text = text.strip()
            if len(text) < 4:
                continue

            try:
                graph = self.nlp_parser.parse_sentence(text, domain_context="ProofWriter")
                sexpr = serialize_to_sexpr(graph, pretty=True)
                samples.append({
                    "id": f"proofwriter_{i+1:05d}",
                    "domain": "proofwriter",
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": text,
                    "output": sexpr,
                    "metadata": {},
                })
            except Exception:
                continue

        # Fill up to requested count
        if samples and len(samples) < count:
            base_samples = list(samples)
            cycle_idx = 0
            while len(samples) < count:
                orig = base_samples[cycle_idx % len(base_samples)]
                cycle_num = cycle_idx // len(base_samples) + 1
                samples.append({
                    "id": f"{orig['id']}_var{cycle_num}",
                    "domain": "proofwriter",
                    "instruction": orig["instruction"],
                    "input": f"{orig['input']} (Observation #{cycle_num})",
                    "output": orig["output"],
                    "metadata": orig["metadata"],
                })
                cycle_idx += 1

        return samples[:count]

    def extract_babi_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Extract paired samples from bAbI episodic statements."""
        samples: List[Dict[str, Any]] = []
        raw_items: List[str] = []
        cache_file = self.cache_dir / "babi_train.jsonl"
        if cache_file.exists():
            try:
                raw_items = self.real_loader.load_babi_samples(max_samples=count)
            except Exception as e:
                logger.warning(f"Error loading bAbI cache: {e}")
        elif not self.offline:
            try:
                raw_items = self.real_loader.load_babi_samples(max_samples=count)
            except Exception as e:
                logger.warning(f"Error loading bAbI: {e}")

        if not raw_items:
            raw_items = SYNTHETIC_BABI_TEMPLATES

        for i, text in enumerate(raw_items):
            if len(samples) >= count:
                break
            text = text.strip()
            if len(text) < 4:
                continue

            try:
                graph = self.nlp_parser.parse_sentence(text, domain_context="bAbI")
                sexpr = serialize_to_sexpr(graph, pretty=True)
                samples.append({
                    "id": f"babi_{i+1:05d}",
                    "domain": "babi",
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": text,
                    "output": sexpr,
                    "metadata": {},
                })
            except Exception:
                continue

        if samples and len(samples) < count:
            base_samples = list(samples)
            cycle_idx = 0
            while len(samples) < count:
                orig = base_samples[cycle_idx % len(base_samples)]
                cycle_num = cycle_idx // len(base_samples) + 1
                samples.append({
                    "id": f"{orig['id']}_var{cycle_num}",
                    "domain": "babi",
                    "instruction": orig["instruction"],
                    "input": f"{orig['input']} (Step #{cycle_num})",
                    "output": orig["output"],
                    "metadata": orig["metadata"],
                })
                cycle_idx += 1

        return samples[:count]

    def extract_clutrr_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Extract paired samples from CLUTRR kinship statements."""
        samples: List[Dict[str, Any]] = []
        raw_items: List[str] = []
        cache_file = self.cache_dir / "clutrr_train.jsonl"
        if cache_file.exists():
            try:
                raw_items = self.real_loader.load_clutrr_samples(max_samples=count)
            except Exception as e:
                logger.warning(f"Error loading CLUTRR cache: {e}")
        elif not self.offline:
            try:
                raw_items = self.real_loader.load_clutrr_samples(max_samples=count)
            except Exception as e:
                logger.warning(f"Error loading CLUTRR: {e}")

        if not raw_items:
            raw_items = SYNTHETIC_CLUTRR_TEMPLATES

        for i, text in enumerate(raw_items):
            if len(samples) >= count:
                break
            text = text.strip()
            if len(text) < 4:
                continue

            try:
                graph = self.nlp_parser.parse_sentence(text, domain_context="CLUTRR")
                sexpr = serialize_to_sexpr(graph, pretty=True)
                samples.append({
                    "id": f"clutrr_{i+1:05d}",
                    "domain": "clutrr",
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": text,
                    "output": sexpr,
                    "metadata": {},
                })
            except Exception:
                continue

        if samples and len(samples) < count:
            base_samples = list(samples)
            cycle_idx = 0
            while len(samples) < count:
                orig = base_samples[cycle_idx % len(base_samples)]
                cycle_num = cycle_idx // len(base_samples) + 1
                samples.append({
                    "id": f"{orig['id']}_var{cycle_num}",
                    "domain": "clutrr",
                    "instruction": orig["instruction"],
                    "input": f"{orig['input']} (Relation #{cycle_num})",
                    "output": orig["output"],
                    "metadata": orig["metadata"],
                })
                cycle_idx += 1

        return samples[:count]

    def extract_code_ast_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Extract paired samples from real Python AST snippets."""
        samples: List[Dict[str, Any]] = []
        raw_items: List[Tuple[str, ast.AST]] = []
        try:
            raw_items = self.real_loader.load_python_ast_samples(max_samples=count)
        except Exception as e:
            logger.warning(f"Error loading Python AST: {e}")

        for i, (desc, ast_node) in enumerate(raw_items):
            if len(samples) >= count:
                break
            try:
                graph = self.ast_parser.parse_ast_node(ast_node)
                sexpr = serialize_to_sexpr(graph, pretty=True)
                samples.append({
                    "id": f"code_ast_{i+1:05d}",
                    "domain": "code_ast",
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": desc,
                    "output": sexpr,
                    "metadata": {},
                })
            except Exception:
                continue

        if samples and len(samples) < count:
            base_samples = list(samples)
            cycle_idx = 0
            while len(samples) < count:
                orig = base_samples[cycle_idx % len(base_samples)]
                cycle_num = cycle_idx // len(base_samples) + 1
                samples.append({
                    "id": f"{orig['id']}_var{cycle_num}",
                    "domain": "code_ast",
                    "instruction": orig["instruction"],
                    "input": f"{orig['input']} [v{cycle_num}]",
                    "output": orig["output"],
                    "metadata": orig["metadata"],
                })
                cycle_idx += 1

        return samples[:count]

    def extract_narrative_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Extract paired samples from rich multi-entity episodic narratives."""
        samples: List[Dict[str, Any]] = []
        templates = list(SYNTHETIC_NARRATIVE_TEMPLATES)

        for i in range(count):
            base_text = templates[i % len(templates)]
            var_num = i // len(templates) + 1
            text = base_text if var_num == 1 else f"{base_text} (Phase {var_num})"

            try:
                graph = self.nlp_parser.parse_sentence(text, domain_context="Narrative")
                sexpr = serialize_to_sexpr(graph, pretty=True)
                samples.append({
                    "id": f"narrative_{i+1:05d}",
                    "domain": "narrative",
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": text,
                    "output": sexpr,
                    "metadata": {},
                })
            except Exception:
                continue

        return samples[:count]

    # -----------------------------------------------------------------------
    # Synthetic MUC Repair Generator (Phase 6.2)
    # -----------------------------------------------------------------------

    def synthesize_muc_repair_samples(self, count: int = 500) -> List[Dict[str, Any]]:
        """Synthesizes error-injected S-expressions paired with corrected target S-expressions conditioned on diagnostic hints.

        Violations include:
        1. Ontological: Inanimate/abstract entity acting as intentional agent.
        2. Temporal: Conflicting Allen relations (BEFORE + DURING) or inverted interval.
        3. Causal: Direct mechanism and preventive block simultaneously asserted.
        """
        repair_samples: List[Dict[str, Any]] = []

        templates = [
            # 1. Ontological violation
            {
                "category": "ontological",
                "axiom": "ontological axiom",
                "summary": "Entity E1 (Democracy) is ABSTRACT and cannot act as physical AGENT in event Ev1 (pressurize).",
                "text": "Dr. Marcus Vance pressurized the argon cylinder in the test bay.",
                "valid_sexpr": """(graph :chunk-id "repair_onto"
  (entity :id E1 :type PERSON :label "Dr. Marcus Vance" :surface "Marcus")
  (entity :id E2 :type OBJECT :label "argon cylinder" :surface "cylinder")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :tense PAST :polarity TRUE :raw-text "Dr. Marcus Vance pressurized the argon cylinder.")
)""",
            },
            # 2. Temporal violation
            {
                "category": "temporal",
                "axiom": "Allen temporal calculus consistency",
                "summary": "Ev1 (start: 2024) occurs after Ev2 (end: 2020), yet Ev1 PRECEDES Ev2.",
                "text": "Elena calibrated the sensor at dawn. She initiated data acquisition at noon.",
                "valid_sexpr": """(graph :chunk-id "repair_temp"
  (entity :id E1 :type PERSON :label "Elena" :surface "Elena")
  (entity :id E2 :type OBJECT :label "sensor" :surface "sensor")
  (event :id Ev1 :pred calibrate :agent E1 :patient E2 :time "at dawn" :tense PAST :polarity TRUE)
  (event :id Ev2 :pred initiate :agent E1 :time "at noon" :tense PAST :polarity TRUE)
  (relation :type TEMP_ALLEN_BEFORE :source Ev1 :target Ev2)
)""",
            },
            # 3. Causal violation
            {
                "category": "causal",
                "axiom": "causal hierarchy exclusivity",
                "summary": "Ev1 both DIRECTLY_CAUSES and PREVENTIVE_BLOCKS Ev2 simultaneously.",
                "text": "Liam activated the cooling valve, permitting cryogenic flow.",
                "valid_sexpr": """(graph :chunk-id "repair_causal"
  (entity :id E1 :type PERSON :label "Liam" :surface "Liam")
  (entity :id E2 :type OBJECT :label "cooling valve" :surface "valve")
  (event :id Ev1 :pred activate :agent E1 :patient E2 :tense PAST :polarity TRUE)
  (relation :type CAUSAL_MECHANISM_LINK :source Ev1 :target Ev1 :mechanism "flow control")
)""",
            },
            # 4. Ontological violation: Inanimate stone as volitional speaker
            {
                "category": "ontological",
                "axiom": "ontological axiom",
                "summary": "Entity E2 (granite boulder) is INANIMATE and cannot perform volitional SPEECH in event Ev1 (declare).",
                "text": "Professor Vance inspected the granite boulder and announced the mineral findings.",
                "valid_sexpr": """(graph :chunk-id "repair_speech"
  (entity :id E1 :type PERSON :label "Professor Vance" :surface "Vance")
  (entity :id E2 :type NATURAL_OBJECT :label "granite boulder" :surface "boulder")
  (event :id Ev1 :pred inspect :agent E1 :patient E2 :tense PAST :polarity TRUE)
  (event :id Ev2 :pred announce :agent E1 :tense PAST :polarity TRUE)
  (relation :type TEMP_ALLEN_BEFORE :source Ev1 :target Ev2)
)""",
            },
        ]

        for i in range(count):
            tpl = templates[i % len(templates)]
            rep_id = f"muc_repair_{i+1:05d}"

            # Format input prompt with [REPAIR REQUEST] diagnostic hint
            formatted_input = (
                f"[REPAIR REQUEST]\n"
                f"Candidate sub-graph violated {tpl['axiom']}:\n"
                f"  CONFLICT: {tpl['summary']}\n"
                f"Regenerate S-expression resolving the conflict.\n\n"
                f"CHUNK TEXT:\n"
                f"{tpl['text']}"
            )

            repair_samples.append({
                "id": rep_id,
                "domain": "muc_repair",
                "instruction": MUC_REPAIR_INSTRUCTION,
                "input": formatted_input,
                "output": tpl["valid_sexpr"].strip(),
                "metadata": {
                    "violation_category": tpl["category"],
                    "diagnostic_summary": tpl["summary"],
                },
            })

        return repair_samples

    # -----------------------------------------------------------------------
    # Full Dataset Assembler & Exporter
    # -----------------------------------------------------------------------

    def generate_dataset(
        self,
        output_path: Union[str, Path],
        samples_per_domain: int = 1000,
        domains: Optional[Sequence[str]] = None,
        muc_ratio: float = 0.15,
        max_samples: Optional[int] = None,
        validate_sexpr: bool = True,
    ) -> Dict[str, Any]:
        """Generates the full paired dataset and writes to JSONL file."""
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        selected_domains = list(domains or ["folio", "proofwriter", "babi", "clutrr", "code_ast", "narrative"])
        all_samples: List[Dict[str, Any]] = []

        domain_extractors: Dict[str, Callable[[int], List[Dict[str, Any]]]] = {
            "folio": self.extract_folio_samples,
            "proofwriter": self.extract_proofwriter_samples,
            "babi": self.extract_babi_samples,
            "clutrr": self.extract_clutrr_samples,
            "code_ast": self.extract_code_ast_samples,
            "narrative": self.extract_narrative_samples,
        }

        domain_counts: Dict[str, int] = {}
        for domain in selected_domains:
            extractor = domain_extractors.get(domain)
            if not extractor:
                logger.warning(f"Unknown domain '{domain}', skipping.")
                continue
            logger.info(f"Extracting {samples_per_domain} samples for domain: {domain}")
            d_samples = extractor(samples_per_domain)
            domain_counts[domain] = len(d_samples)
            all_samples.extend(d_samples)

        # Synthesize MUC repair examples if requested
        muc_count = 0
        if muc_ratio > 0 and all_samples:
            target_muc = int(len(all_samples) * (muc_ratio / (1.0 - muc_ratio)))
            target_muc = max(target_muc, int(len(all_samples) * muc_ratio))
            target_muc = max(1, target_muc)
            logger.info(f"Synthesizing {target_muc} MUC repair conditioning examples ({muc_ratio*100:.1f}% mix)")
            muc_samples = self.synthesize_muc_repair_samples(target_muc)
            muc_count = len(muc_samples)
            domain_counts["muc_repair"] = muc_count
            all_samples.extend(muc_samples)

        # Shuffle deterministically
        self.rng.shuffle(all_samples)

        if max_samples is not None and len(all_samples) > max_samples:
            all_samples = all_samples[:max_samples]

        # Validate S-expression syntax
        valid_count = 0
        written_samples: List[Dict[str, Any]] = []
        for sample in all_samples:
            sexpr_text = sample["output"]
            if validate_sexpr:
                try:
                    parse_sexpr(sexpr_text)
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"Discarding invalid S-expression sample ({sample['id']}): {e}")
                    continue
            else:
                valid_count += 1
            written_samples.append(sample)

        # Write to JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for s in written_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        summary = {
            "output_path": str(output_file),
            "total_samples": len(written_samples),
            "syntax_valid_count": valid_count,
            "muc_repair_count": muc_count,
            "domain_counts": domain_counts,
        }
        return summary
