"""Natural Language Generation (NLG) Realizer for QUANTA.

Deterministically unrolls Content-Addressed Abstract Syntax Graphs (ASGs)
into grammatical, fluent English sentences.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_name
from core.types import QuantaVector, QuaternaryValue


class EnglishRealizer:
    """Deterministic English NLG realizer unrolling QuantaGraph ASGs to natural text."""

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
        """Realizes an entire QuantaGraph ASG into an English sentence string."""
        root = graph.root
        if root is None:
            return ""

        # Check for conditional / implicational sentences
        if root.get_slot("LJB_GANAI_IF_THEN") == 1 or root.get_slot("GRAPH_BRANCH_COND") == 1:
            return self._realize_conditional(graph, root)

        # Check if root has sub-expressions
        if "GRAPH_IS_SUB_EXP" in root.edges and len(root.edges["GRAPH_IS_SUB_EXP"]) > 0:
            sub_clauses = []
            for child_cid in root.edges["GRAPH_IS_SUB_EXP"]:
                child_node = graph.get_node(child_cid)
                if child_node:
                    sub_clauses.append(self._realize_clause(graph, child_node))
            if sub_clauses:
                connective = " and " if root.get_slot("LJB_JE_AND") == 1 else (" or " if root.get_slot("LJB_JA_OR") == 1 else ", ")
                return connective.join(sub_clauses).strip().capitalize() + "."

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
            is_permission=is_permission,
            is_possibility=is_possibility,
            is_probable=is_probable,
            subject=agent_str,
        )

        # 4. Resolve Patient / Object (VAL_X2_PATIENT) or Experiencer
        patient_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X2_PATIENT")
        experiencer_str = ""
        if "VAL_EXPERIENCER" in predicate_node.edges:
            if patient_str:
                experiencer_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_EXPERIENCER", prep="to")
            else:
                patient_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_EXPERIENCER")

        # 5. Resolve Prepositional Arguments
        dest_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X3_DESTINATION", prep="to")
        source_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X4_SOURCE", prep="from")
        inst_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_X5_INSTRUMENT", prep="with")
        loc_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_LOCATION_SLOT", prep="in")
        purpose_str = self._resolve_entity_by_edge(graph, predicate_node, "VAL_PURPOSE_SLOT", prep="for")

        # 6. Resolve Temporal & Manner Adverbials
        time_str = self._resolve_temporal_phrase(predicate_node)
        manner_str = self._resolve_manner_phrase(predicate_node)

        # Assemble clause tokens in canonical SVO order
        tokens = [agent_str] if agent_str else []
        if verb_phrase:
            tokens.append(verb_phrase)
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
        if node.anchor and node.anchor.startswith("wn:"):
            # e.g., 'wn:bite.v.01' -> 'bite'
            parts = node.anchor[3:].split(".")
            if len(parts) > 0 and parts[0]:
                return parts[0].replace("_", " ")

        if node.literal and isinstance(node.literal, str):
            lit = node.literal.strip()
            # If literal is a simple verb or short text
            if len(lit.split()) == 1 and lit.isalpha():
                return lit.lower()

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
    ) -> str:
        """Constructs an inflected verb phrase with tense, modals, and negation."""
        is_copula = verb_base in ("be", "is", "are")
        
        # Modal auxiliary construction
        if is_obligation:
            modal = "must not" if is_negated else "must"
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

        if node.anchor and node.anchor.startswith("wn:"):
            # e.g., 'wn:golden_retriever.n.01' -> 'golden retriever'
            parts = node.anchor[3:].split(".")
            if len(parts) > 0:
                head_noun = parts[0].replace("_", " ")

        if not head_noun and node.literal:
            lit = str(node.literal).strip()
            # If literal is a clean entity name
            head_noun = lit

        if not head_noun:
            if node.get_slot("TYPE_HUMAN") == 1:
                head_noun = "person"
            elif node.get_slot("TYPE_ANIMATE") == 1:
                head_noun = "animal"
            elif node.get_slot("TYPE_ARTIFACT") == 1:
                head_noun = "object"
            elif node.get_slot("TYPE_SPATIAL_REGION") == 1:
                head_noun = "place"
            elif node.get_slot("TYPE_ABSTRACT_CONCEPT") == 1:
                head_noun = "concept"
            else:
                head_noun = "entity"

        # Check if head noun is a proper noun or pronoun
        if head_noun.istitle() and len(head_noun.split()) == 1 and head_noun.lower() not in ("person", "animal", "dog", "cat", "mailman"):
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
            determiner = "this"
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
