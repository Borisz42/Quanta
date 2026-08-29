"""Real benchmark dataset loader and forward converter for FOLIO, ProofWriter, bAbI, CLUTRR, and Code ASTs."""

from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import requests

from core.asg import QuantaGraph, QuantaNode
from core.slots import SLOT_NAME_TO_INDEX
from core.types import QuantaVector
from data.corpus_generator import GeneratedCorpus
from parser.ast_parser import ASTForwardParser
from parser.fol_parser import FOLParser
from parser.nlp_forward import NLPForwardParser
from profiler.candidate_pool import build_candidate_pool, project_canonical_to_candidates


class RealDatasetLoader:
    """Downloads, saves raw datasets locally, and converts them into QUANTA Mentalese representations."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir or "data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.nlp_parser = NLPForwardParser()
        self.fol_parser = FOLParser()
        self.ast_parser = ASTForwardParser()
        self.candidates = build_candidate_pool()
        self.cand_name_to_idx = {c.name: c.id for c in self.candidates}

    def download_all_raw_datasets(self, samples_to_cache: int = 5000):
        """Ensures all 4 raw benchmark datasets are downloaded and stored locally in data/raw/."""
        print(f"Checking / downloading all raw benchmark datasets to {self.cache_dir.resolve()}...")
        self.load_folio_samples(max_samples=samples_to_cache)
        self.load_proofwriter_samples(max_samples=samples_to_cache)
        self.load_babi_samples(max_samples=samples_to_cache)
        self.load_clutrr_samples(max_samples=samples_to_cache)
        print("All raw benchmark datasets are cached locally.")

    def load_folio_samples(self, max_samples: int = 1000) -> List[Tuple[str, str]]:
        """Downloads and loads distinct FOLIO premise sentences and their matching First-Order Logic formulas."""
        cache_file = self.cache_dir / "folio_train.jsonl"
        val_cache = self.cache_dir / "folio_val.jsonl"

        if not cache_file.exists():
            print("  Downloading raw FOLIO train dataset...")
            url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/master/data/v0.0/folio-train.jsonl"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)

        if not val_cache.exists():
            val_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/master/data/v0.0/folio-validation.jsonl"
            resp = requests.get(val_url, timeout=30)
            if resp.status_code == 200:
                with open(val_cache, "w", encoding="utf-8") as f:
                    f.write(resp.text)

        samples: List[Tuple[str, str]] = []
        seen_texts: Set[str] = set()

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    raw_premises = item.get("premises", "")
                    raw_fol = item.get("premises-FOL", "")

                    if isinstance(raw_premises, list) and isinstance(raw_fol, list):
                        for p_text, p_fol in zip(raw_premises, raw_fol):
                            p_clean = str(p_text).strip()
                            if len(p_clean) > 3 and p_clean not in seen_texts:
                                seen_texts.add(p_clean)
                                samples.append((p_clean, str(p_fol).strip()))
                                if len(samples) >= max_samples:
                                    break
                    else:
                        premise_list = raw_premises if isinstance(raw_premises, list) else str(raw_premises).split(".")
                        for sent in premise_list:
                            s_str = str(sent).strip()
                            if len(s_str) > 3 and s_str not in seen_texts:
                                seen_texts.add(s_str)
                                samples.append((s_str, ""))
                                if len(samples) >= max_samples:
                                    break

                    # Also include conclusion
                    conc_text = str(item.get("conclusion", "")).strip()
                    conc_fol = str(item.get("conclusion-FOL", "")).strip()
                    if len(conc_text) > 3 and conc_text not in seen_texts:
                        seen_texts.add(conc_text)
                        samples.append((conc_text, conc_fol))

                    if len(samples) >= max_samples:
                        break
        return samples[:max_samples]

    def load_proofwriter_samples(self, max_samples: int = 1000) -> List[str]:
        """Loads distinct ProofWriter theory facts, rules, and questions."""
        cache_file = self.cache_dir / "proofwriter_train.jsonl"
        if not cache_file.exists():
            try:
                from datasets import load_dataset
                ds = load_dataset("tasksource/proofwriter", split="train", streaming=True)
                count = 0
                with open(cache_file, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                        count += 1
                        if count >= max(max_samples * 2, 5000):
                            break
            except Exception as e:
                print(f"  Warning: failed to download ProofWriter from HF: {e}")

        samples: List[str] = []
        seen_texts: Set[str] = set()

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    theory = item.get("theory", "")
                    question = item.get("question", "")

                    for rule_or_fact in str(theory).split("."):
                        cleaned = rule_or_fact.strip()
                        if len(cleaned) > 3 and cleaned not in seen_texts:
                            seen_texts.add(cleaned)
                            samples.append(cleaned)
                            if len(samples) >= max_samples:
                                break

                    if question and question not in seen_texts:
                        seen_texts.add(question)
                        samples.append(question)

                    if len(samples) >= max_samples:
                        break
        return samples[:max_samples]

    def load_babi_samples(self, max_samples: int = 1000) -> List[str]:
        """Loads distinct bAbI spatial, movement, and transfer statements."""
        cache_file = self.cache_dir / "babi_train.jsonl"
        if not cache_file.exists():
            try:
                from datasets import load_dataset
                ds = load_dataset("Muennighoff/babi", split="train", streaming=True)
                count = 0
                with open(cache_file, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                        count += 1
                        if count >= max(max_samples * 2, 5000):
                            break
            except Exception as e:
                print(f"  Warning: failed to download bAbI from HF: {e}")

        samples: List[str] = []
        seen_texts: Set[str] = set()

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    passage = item.get("passage", "")
                    for sent in str(passage).split("\n"):
                        cleaned = sent.strip()
                        if cleaned and not cleaned.isdigit():
                            if cleaned[0].isdigit():
                                cleaned = cleaned.split(" ", 1)[-1]
                            if len(cleaned) > 3 and cleaned not in seen_texts:
                                seen_texts.add(cleaned)
                                samples.append(cleaned)
                                if len(samples) >= max_samples:
                                    break
                    if len(samples) >= max_samples:
                        break
        return samples[:max_samples]

    def load_clutrr_samples(self, max_samples: int = 1000) -> List[str]:
        """Loads distinct CLUTRR kinship statements."""
        cache_file = self.cache_dir / "clutrr_train.jsonl"
        if not cache_file.exists():
            try:
                from datasets import load_dataset
                ds = load_dataset("tasksource/clutrr", split="train", streaming=True)
                count = 0
                with open(cache_file, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                        count += 1
                        if count >= max(max_samples * 2, 5000):
                            break
            except Exception as e:
                print(f"  Warning: failed to download CLUTRR from HF: {e}")

        samples: List[str] = []
        seen_texts: Set[str] = set()

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    story = item.get("sentence1", "")
                    cleaned_story = str(story).replace("[", "").replace("]", "")
                    for sent in cleaned_story.split("."):
                        s_clean = sent.strip()
                        if len(s_clean) > 3 and s_clean not in seen_texts:
                            seen_texts.add(s_clean)
                            samples.append(s_clean)
                            if len(samples) >= max_samples:
                                break
                    if len(samples) >= max_samples:
                        break
        return samples[:max_samples]

    def load_python_ast_samples(self, max_samples: int = 1000) -> List[Tuple[str, ast.AST]]:
        """Loads and parses real, idiomatic, clean Python code snippets into AST trees."""
        cache_file = self.cache_dir / "python_ast_samples.jsonl"
        samples: List[Tuple[str, ast.AST]] = []

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    code = item.get("code", "")
                    one_liner = item.get("one_liner", "")
                    if not one_liner:
                        one_liner = " ".join(code.strip().splitlines())
                    if len(one_liner) > 90:
                        one_liner = one_liner[:87] + "..."
                    
                    try:
                        tree = ast.parse(code.strip())
                        if tree.body:
                            desc = f"CodeAST: {one_liner}"
                            samples.append((desc, tree.body[0]))
                    except Exception:
                        continue

        # Fallback if cache file is empty or not found
        if not samples:
            py_files = list(Path("src").rglob("*.py"))
            for py_file in py_files:
                try:
                    with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                        tree = ast.parse(f.read())
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.If, ast.For, ast.While)):
                            code_str = ast.unparse(node).replace("\n", " ")
                            if len(code_str) > 90:
                                code_str = code_str[:87] + "..."
                            desc = f"CodeAST: {code_str}"
                            samples.append((desc, node))
                except Exception:
                    continue

        if samples and len(samples) < max_samples:
            base_samples = list(samples)
            idx = 0
            while len(samples) < max_samples:
                desc, node = base_samples[idx % len(base_samples)]
                dup_num = idx // len(base_samples) + 1
                samples.append((f"{desc} [var #{dup_num}]", node))
                idx += 1

        return samples[:max_samples]

    def build_real_corpus(self, samples_per_domain: int = 1000) -> GeneratedCorpus:
        """Builds a balanced multi-domain real reasoning corpus parsed into QUANTA vectors."""
        canonical_rows: List[np.ndarray] = []
        labels: List[int] = []
        propositions: List[str] = []

        # 1. FOLIO (Domain 0)
        print(f"  Ingesting {samples_per_domain} real FOLIO propositions...")
        folio_items = self.load_folio_samples(max_samples=samples_per_domain)
        for sent, fol in folio_items:
            try:
                if fol:
                    graph = self.fol_parser.parse_formula(fol)
                else:
                    graph = self.nlp_parser.parse_sentence(sent, domain_context="FOLIO")
                vec = graph.to_proposition_vector().to_numpy()
            except Exception:
                vec = self.nlp_parser.sentence_to_vector(sent, domain_context="FOLIO").to_numpy()

            canonical_rows.append(vec)
            labels.append(0)
            propositions.append(f"FOLIO: {sent}")

        # 2. ProofWriter (Domain 1)
        print(f"  Ingesting {samples_per_domain} real ProofWriter propositions...")
        pw_items = self.load_proofwriter_samples(max_samples=samples_per_domain)
        for sent in pw_items:
            vec = self.nlp_parser.sentence_to_vector(sent, domain_context="ProofWriter").to_numpy()
            canonical_rows.append(vec)
            labels.append(1)
            propositions.append(f"ProofWriter: {sent}")

        # 3. bAbI (Domain 2)
        print(f"  Ingesting {samples_per_domain} real bAbI propositions...")
        babi_items = self.load_babi_samples(max_samples=samples_per_domain)
        for sent in babi_items:
            vec = self.nlp_parser.sentence_to_vector(sent, domain_context="bAbI").to_numpy()
            canonical_rows.append(vec)
            labels.append(2)
            propositions.append(f"bAbI: {sent}")

        # 4. CLUTRR (Domain 3)
        print(f"  Ingesting {samples_per_domain} real CLUTRR propositions...")
        clutrr_items = self.load_clutrr_samples(max_samples=samples_per_domain)
        for sent in clutrr_items:
            vec = self.nlp_parser.sentence_to_vector(sent, domain_context="CLUTRR").to_numpy()
            canonical_rows.append(vec)
            labels.append(3)
            propositions.append(f"CLUTRR: {sent}")

        canonical_matrix = np.stack(canonical_rows, axis=0)
        candidate_matrix = project_canonical_to_candidates(canonical_matrix, self.candidates)

        return GeneratedCorpus(
            canonical_matrix=canonical_matrix,
            candidate_matrix=candidate_matrix,
            labels=np.array(labels, dtype=np.int32),
            propositions=propositions,
        )
