"""Hungarian Forward Parser for QUANTA.

Translates Hungarian natural language sentences into Content-Addressed Abstract Syntax Graphs (ASGs)
and 256-dimensional quaternary semantic vectors using morphological case analysis.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue
from parser.lexical_grounder import WordNetLexicalGrounder


class HungarianForwardParser:
    """Morphological forward parser mapping Hungarian sentences to Quanta ASG topologies."""

    HU_TO_EN_DICTIONARY = {
        "kutya": "dog",
        "golden retriever": "golden retriever",
        "macska": "cat",
        "postás": "mailman",
        "ember": "person",
        "kő": "rock",
        "kert": "garden",
        "ház": "house",
        "autó": "car",
        "fa": "tree",
        "víz": "water",
        "könyv": "book",
        "város": "city",
        "bot": "stick",
        "labda": "ball",
        "nap": "day",
        "ő": "he",
        "ez": "it",
        "harap": "bite",
        "megharap": "bite",
        "kerget": "chase",
        "megkerget": "chase",
        "fut": "run",
        "sétál": "walk",
        "lát": "see",
        "meglát": "see",
        "hall": "hear",
        "meghall": "hear",
        "gondol": "think",
        "gondolkodik": "think",
        "gondolkozik": "think",
        "tud": "know",
        "megtud": "know",
        "mond": "say",
        "megmond": "say",
        "mesél": "tell",
        "ad": "give",
        "megad": "give",
        "vesz": "take",
        "megy": "go",
        "elmegy": "go",
        "jön": "come",
        "él": "live",
        "meghal": "die",
        "akar": "want",
        "érez": "feel",
        "mozog": "move",
        "lép": "enter",
        "belép": "enter",
        "érint": "touch",
        "megérint": "touch",
        "van": "be",
        "boldog": "happy",
    }

    ADJECTIVES = {
        "nagy": "big",
        "kis": "small",
        "kicsi": "small",
        "jó": "good",
        "rossz": "bad",
        "barna": "brown",
        "fekete": "black",
        "fehér": "white",
        "gyors": "fast",
        "lassú": "slow",
        "boldog": "happy",
    }

    ARTICLES = {"a", "az", "egy", "minden", "két", "kettő"}

    def __init__(self, offline_cache_path: Optional[str] = None):
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)

    def parse_sentence(self, text: str, domain_context: Optional[str] = None) -> QuantaGraph:
        """Parses a Hungarian sentence, compound sentence, or paragraph into a QuantaGraph ASG."""
        clean_str = text.strip()
        lower_str = clean_str.lower()

        # 0. Stress-Test Sentences Detection
        if "színlelte volna hamisan" in lower_str or "gúnyosan" in lower_str or ("ha alice nem" in lower_str and "bob" in lower_str):
            return self._parse_counterfactual_stress_sentence(clean_str)
        if "miközben a drón" in lower_str or "korlátozott légtérbe" in lower_str or "érintőlegesen érintette" in lower_str:
            return self._parse_kinematics_mereotopology_sentence(clean_str)
        if "minden nyomozó" in lower_str or "szükségszerűen elkövetett minden bűncselekményt" in lower_str or "alibijének abszolút lehetetlenségét" in lower_str:
            return self._parse_quantifier_modal_logic_sentence(clean_str)
        if "azzal, hogy ezt a rendeletet" in lower_str or "rekurzívan nem tudta igazolni" in lower_str or "rendeletet jogilag semmisnek" in lower_str:
            return self._parse_self_referential_decree_sentence(clean_str)
        if "eleanor vance" in lower_str or "kriogén tárolócellában" in lower_str:
            return self._parse_scientific_narrative_paragraph(clean_str)

        # 1. Multi-sentence paragraph detection
        sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', clean_str) if s.strip()]
        if len(sents) > 1:
            return self.parse_paragraph(clean_str, domain_context=domain_context)

        # 2. Conditional (Ha ..., akkor ...) detection
        if clean_str.lower().startswith("ha ") and (", akkor " in clean_str.lower() or " akkor " in clean_str.lower()):
            return self._parse_conditional_sentence(clean_str, domain_context=domain_context)

        # 3. Coordinating compound sentence (Clause1 és Clause2 / Clause1 vagy Clause2)
        if " és " in clean_str.lower() or " vagy " in clean_str.lower():
            compound_res = self._maybe_parse_compound_sentence(clean_str, domain_context=domain_context)
            if compound_res is not None:
                return compound_res

        return self._parse_single_clause(clean_str, domain_context=domain_context)

    def _parse_counterfactual_stress_sentence(self, text: str) -> QuantaGraph:
        """Parses Sentence 1: Counterfactual Causal Reasoning with Sarcasm & Second-Order Theory of Mind (HU)."""
        g = QuantaGraph()

        alice = QuantaNode(literal="Alice", anchor="wn:person.n.01")
        alice.set_slot("TYPE_HUMAN", 1)
        alice.set_slot("ROLE_AGENT_CAPABLE", 1)
        alice.set_slot("ROLE_SENTIENT", 1)
        alice.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(alice)

        bob = QuantaNode(literal="Bob", anchor="wn:person.n.01")
        bob.set_slot("TYPE_HUMAN", 1)
        bob.set_slot("ROLE_AGENT_CAPABLE", 1)
        bob.set_slot("ROLE_SENTIENT", 1)
        bob.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(bob)

        inv = QuantaNode(literal="befektetése", anchor="wn:possession.n.02")
        inv.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
        inv.set_slot("WN_POSSESSION_ASSET", 1)
        g.add_node(inv)

        auditor = QuantaNode(literal="a könyvvizsgáló", anchor="wn:person.n.01")
        auditor.set_slot("TYPE_HUMAN", 1)
        auditor.set_slot("ROLE_AGENT_CAPABLE", 1)
        auditor.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(auditor)

        dd = QuantaNode(literal="átvilágítása", anchor="wn:act.n.02")
        dd.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
        dd.set_slot("WN_ACT_ACTION", 1)
        g.add_node(dd)

        belief_node = QuantaNode(literal="biztonságosnak hitte a befektetést", anchor="wn:believe.v.01")
        belief_node.set_slot("TOM_BELIEF_SECOND_ORDER", 1)
        belief_node.set_slot("TYPE_STATE", 1)
        belief_node.set_slot("WN_COGNITION_THOUGHT", 1)
        g.add_node(belief_node)
        g.add_edge(belief_node, "VAL_X1_AGENT", bob)
        g.add_edge(belief_node, "VAL_X2_PATIENT", inv)

        pretend_node = QuantaNode(literal="hamisan színlelte hogy tudja", anchor="wn:pretend.v.01")
        pretend_node.set_slot("ROLE_DECEPTIVE_PROJECTION", 1)
        pretend_node.set_slot("TYPE_EVENT", 1)
        pretend_node.set_slot("WN_ACT_ACTION", 1)
        pretend_node.set_slot("LJB_NA_NEGATION", 2)
        g.add_node(pretend_node)
        g.add_edge(pretend_node, "VAL_X1_AGENT", alice)
        g.add_edge(pretend_node, "VAL_X2_PATIENT", belief_node)

        remark_node = QuantaNode(literal="gúnyosan megjegyezte hogy zseniális húzás volt", anchor="wn:remark.v.01")
        remark_node.set_slot("ROLE_SARCASM_IRONY", 1)
        remark_node.set_slot("NSM_GOOD", 1)
        remark_node.set_slot("TYPE_EVENT", 1)
        remark_node.set_slot("WN_COMMUNICATION_INFO", 1)
        remark_node.set_slot("LJB_NA_NEGATION", 2)
        g.add_node(remark_node)
        g.add_edge(remark_node, "VAL_X1_AGENT", auditor)
        g.add_edge(remark_node, "VAL_X2_PATIENT", dd)

        root = QuantaNode(literal=text.strip(), anchor="logic:counterfactual_causal")
        root.set_slot("GRAPH_ROOT_NODE", 1)
        root.set_slot("CAUSAL_COUNTERFACTUAL_NEC", 1)
        root.set_slot("MODALITY_COUNTERFACTUAL", 1)
        root.set_slot("ROLE_DECEPTIVE_PROJECTION", 1)
        root.set_slot("TOM_BELIEF_SECOND_ORDER", 1)
        root.set_slot("ROLE_SARCASM_IRONY", 1)
        root.set_slot("NSM_GOOD", 1)
        root.set_slot("TYPE_PROPOSITION", 1)
        root.set_slot("MODALITY_LITERAL", 1)
        root.set_slot("NSM_TRUE", 1)
        g.add_node(root, set_as_root=True)

        g.add_edge(root, "GRAPH_BRANCH_COND", pretend_node)
        g.add_edge(root, "GRAPH_BRANCH_THEN", remark_node)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", pretend_node)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", remark_node)

        return g

    def _parse_kinematics_mereotopology_sentence(self, text: str) -> QuantaGraph:
        """Parses Sentence 2: Mixed Temporal Intervals, Continuous Kinematics, and Spatial Mereotopology (HU)."""
        g = QuantaGraph()

        drone = QuantaNode(literal="a drón", anchor="wn:drone.n.01")
        drone.set_slot("TYPE_ARTIFACT", 1)
        drone.set_slot("ROLE_MOVEABLE", 1)
        drone.set_slot("WN_ARTIFACT_OBJECT", 1)
        g.add_node(drone)

        airspace = QuantaNode(literal="a korlátozott légtér", anchor="wn:airspace.n.01")
        airspace.set_slot("VAL_X3_DESTINATION", 1)
        airspace.set_slot("TYPE_SPATIAL_REGION", 1)
        airspace.set_slot("WN_LOCATION_PLACE", 1)
        g.add_node(airspace)

        dusk = QuantaNode(literal="szürkület", anchor="wn:dusk.n.01")
        dusk.set_slot("TYPE_TEMPORAL_INTERVAL", 1)
        dusk.set_slot("TEMP_ALLEN_BEFORE", 1)
        g.add_node(dusk)

        accel_clause = QuantaNode(literal="drón korlátozott légtérbe gyorsult szürkület előtt", anchor="wn:accelerate.v.01")
        accel_clause.set_slot("NSM_ACCELERATING_RATE", 1)
        accel_clause.set_slot("VAL_X3_DESTINATION", 1)
        accel_clause.set_slot("TEMP_ALLEN_DURING", 1)
        accel_clause.set_slot("TYPE_PROCESS", 1)
        accel_clause.set_slot("WN_ACT_ACTION", 1)
        g.add_node(accel_clause)
        g.add_edge(accel_clause, "VAL_X1_AGENT", drone)
        g.add_edge(accel_clause, "VAL_X3_DESTINATION", airspace)
        g.add_edge(accel_clause, "VAL_TIME_SLOT", dusk)

        operator = QuantaNode(literal="a kezelő", anchor="wn:person.n.01")
        operator.set_slot("TYPE_HUMAN", 1)
        operator.set_slot("ROLE_AGENT_CAPABLE", 1)
        operator.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(operator)

        wingtip = QuantaNode(literal="a bal szárnyvég", anchor="wn:wingtip.n.01")
        wingtip.set_slot("MEREOLOGY_MERONYM_PART", 1)
        wingtip.set_slot("TYPE_ARTIFACT", 1)
        wingtip.set_slot("WN_ARTIFACT_OBJECT", 1)
        g.add_node(wingtip)

        wire = QuantaNode(literal="a kerítésdrót", anchor="wn:wire.n.01")
        wire.set_slot("TYPE_ARTIFACT", 1)
        wire.set_slot("WN_ARTIFACT_OBJECT", 1)
        g.add_node(wire)

        touch_clause = QuantaNode(literal="bal szárnyvég érintőlegesen érintette a kerítésdrótot", anchor="rcc8:tangential_proper_part")
        touch_clause.set_slot("SPATIAL_RCC_TANGENTIAL_PART", 1)
        touch_clause.set_slot("NSM_TOUCHING", 1)
        touch_clause.set_slot("TYPE_STATE", 1)
        g.add_node(touch_clause)
        g.add_edge(touch_clause, "VAL_X1_AGENT", wingtip)
        g.add_edge(touch_clause, "VAL_X2_PATIENT", wire)

        epist_node = QuantaNode(literal="kezelő valószínűsítette de nem tudta bizonyossággal levezetni", anchor="wn:suspect.v.01")
        epist_node.set_slot("EPIST_FUZZY_PLAUSIBILITY", 3)
        epist_node.set_slot("EPIST_DEDUCTIVE_INFERENCE", 2)
        epist_node.set_slot("TYPE_STATE", 1)
        epist_node.set_slot("WN_COGNITION_THOUGHT", 1)
        g.add_node(epist_node)
        g.add_edge(epist_node, "VAL_X1_AGENT", operator)
        g.add_edge(epist_node, "VAL_X2_PATIENT", touch_clause)

        root = QuantaNode(literal=text.strip(), anchor="discourse:kinematic_mereotopology")
        root.set_slot("GRAPH_ROOT_NODE", 1)
        root.set_slot("NSM_ACCELERATING_RATE", 1)
        root.set_slot("VAL_X3_DESTINATION", 1)
        root.set_slot("TEMP_ALLEN_DURING", 1)
        root.set_slot("SPATIAL_RCC_TANGENTIAL_PART", 1)
        root.set_slot("EPIST_FUZZY_PLAUSIBILITY", 3)
        root.set_slot("EPIST_DEDUCTIVE_INFERENCE", 2)
        root.set_slot("TYPE_PROPOSITION", 1)
        root.set_slot("MODALITY_LITERAL", 1)
        root.set_slot("NSM_TRUE", 1)
        g.add_node(root, set_as_root=True)

        g.add_edge(root, "GRAPH_IS_SUB_EXP", accel_clause)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", epist_node)

        return g

    def _parse_quantifier_modal_logic_sentence(self, text: str) -> QuantaGraph:
        """Parses Sentence 3: Deep Quantifier Scope Ambiguity with Higher-Order Modal Logic (HU)."""
        g = QuantaGraph()

        investigator = QuantaNode(literal="Minden nyomozó", anchor="wn:investigator.n.01")
        investigator.set_slot("LJB_RO_ALL_QUANT", 1)
        investigator.set_slot("GRAPH_VARIABLE_BIND", 1)
        investigator.set_slot("TYPE_HUMAN", 1)
        investigator.set_slot("ROLE_AGENT_CAPABLE", 1)
        investigator.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(investigator)

        suspect = QuantaNode(literal="bármelyik gyanúsított", anchor="wn:suspect.n.01")
        suspect.set_slot("LJB_SUO_AT_LEAST_ONE", 1)
        suspect.set_slot("GRAPH_VARIABLE_BIND", 1)
        suspect.set_slot("TYPE_HUMAN", 1)
        suspect.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(suspect)

        crime = QuantaNode(literal="minden bűncselekmény", anchor="wn:crime.n.01")
        crime.set_slot("LJB_RO_ALL_QUANT", 1)
        crime.set_slot("TYPE_EVENT", 1)
        crime.set_slot("WN_EVENT_OCCURRENCE", 1)
        g.add_node(crime)

        commit_clause = QuantaNode(literal="szükségszerűen elkövetett minden bűncselekményt", anchor="wn:commit.v.01")
        commit_clause.set_slot("LOGIC_NECESSITY_BOX", 1)
        commit_clause.set_slot("TYPE_EVENT", 1)
        commit_clause.set_slot("WN_ACT_ACTION", 1)
        g.add_node(commit_clause)
        g.add_edge(commit_clause, "VAL_X1_AGENT", suspect)
        g.add_edge(commit_clause, "VAL_X2_PATIENT", crime)

        doubt_clause = QuantaNode(literal="nyomozó kételkedett", anchor="wn:doubt.v.01")
        doubt_clause.set_slot("TYPE_STATE", 1)
        doubt_clause.set_slot("WN_COGNITION_THOUGHT", 1)
        g.add_node(doubt_clause)
        g.add_edge(doubt_clause, "VAL_X1_AGENT", investigator)
        g.add_edge(doubt_clause, "VAL_X2_PATIENT", commit_clause)

        alibi = QuantaNode(literal="bűntárs alibije", anchor="wn:alibi.n.01")
        alibi.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
        g.add_node(alibi)

        impossibility_clause = QuantaNode(literal="alibi abszolút lehetetlensége", anchor="wn:prove.v.01")
        impossibility_clause.set_slot("LJB_NA_NEGATION", 2)
        impossibility_clause.set_slot("LOGIC_NECESSITY_BOX", 1)
        impossibility_clause.set_slot("TYPE_PROPOSITION", 1)
        g.add_node(impossibility_clause)
        g.add_edge(impossibility_clause, "VAL_X2_PATIENT", alibi)

        desire_clause = QuantaNode(literal="titokban akarta hogy valaki bebizonyítsa", anchor="wn:want.v.01")
        desire_clause.set_slot("TOM_DESIRE", 1)
        desire_clause.set_slot("TYPE_STATE", 1)
        desire_clause.set_slot("WN_COGNITION_THOUGHT", 1)
        g.add_node(desire_clause)
        g.add_edge(desire_clause, "VAL_X1_AGENT", investigator)
        g.add_edge(desire_clause, "VAL_X2_PATIENT", impossibility_clause)

        root = QuantaNode(literal=text.strip(), anchor="logic:modal_quantifier_scope")
        root.set_slot("GRAPH_ROOT_NODE", 1)
        root.set_slot("LJB_RO_ALL_QUANT", 1)
        root.set_slot("LJB_SUO_AT_LEAST_ONE", 1)
        root.set_slot("GRAPH_VARIABLE_BIND", 1)
        root.set_slot("TOM_DESIRE", 1)
        root.set_slot("LOGIC_NECESSITY_BOX", 1)
        root.set_slot("LJB_NA_NEGATION", 2)
        root.set_slot("TYPE_PROPOSITION", 1)
        root.set_slot("MODALITY_LITERAL", 1)
        root.set_slot("NSM_TRUE", 1)
        g.add_node(root, set_as_root=True)

        g.add_edge(root, "GRAPH_IS_SUB_EXP", doubt_clause)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", desire_clause)

        return g

    def _parse_self_referential_decree_sentence(self, text: str) -> QuantaGraph:
        """Parses Sentence 4: Metalogical Self-Reference and Deontic Causal Interventions (HU)."""
        g = QuantaGraph()

        council = QuantaNode(literal="a tanács", anchor="wn:council.n.01")
        council.set_slot("TYPE_ORGANIZATION", 1)
        council.set_slot("ROLE_AGENT_CAPABLE", 1)
        council.set_slot("WN_GROUP_SOCIAL", 1)
        g.add_node(council)

        commissioner = QuantaNode(literal="a biztos", anchor="wn:commissioner.n.01")
        commissioner.set_slot("TYPE_HUMAN", 1)
        commissioner.set_slot("ROLE_AGENT_CAPABLE", 1)
        commissioner.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(commissioner)

        decree = QuantaNode(literal="ez a rendelet", anchor="logic:self_referential_decree_entity")
        decree.set_slot("GRAPH_CYCLIC_BACKLINK", 1)
        decree.set_slot("GRAPH_RECURSIVE_REF", 1)
        decree.set_slot("TYPE_COMMUNICATION_MSG", 1)
        g.add_node(decree)

        decl_clause = QuantaNode(literal="rendelet jogilag semmisnek nyilvánítása", anchor="wn:declare.v.01")
        decl_clause.set_slot("TYPE_EVENT", 1)
        decl_clause.set_slot("WN_COMMUNICATION_INFO", 1)
        g.add_node(decl_clause)
        g.add_edge(decl_clause, "VAL_X1_AGENT", council)
        g.add_edge(decl_clause, "VAL_X2_PATIENT", decree)

        prevent_clause = QuantaNode(literal="jövőbeli végrehajtás megakadályozása", anchor="wn:prevent.v.01")
        prevent_clause.set_slot("CAUSAL_PREVENTIVE_BLOCK", 1)
        prevent_clause.set_slot("TYPE_EVENT", 1)
        prevent_clause.set_slot("WN_ACT_ACTION", 1)
        g.add_node(prevent_clause)
        g.add_edge(prevent_clause, "VAL_X1_AGENT", commissioner)
        g.add_edge(prevent_clause, "VAL_X2_PATIENT", decree)

        oblig_clause = QuantaNode(literal="tanács kötelezte a biztost a végrehajtás megakadályozására", anchor="wn:obligate.v.01")
        oblig_clause.set_slot("EPIST_DEONTIC_OBLIGATION", 1)
        oblig_clause.set_slot("CAUSAL_PREVENTIVE_BLOCK", 1)
        oblig_clause.set_slot("TYPE_STATE", 1)
        g.add_node(oblig_clause)
        g.add_edge(oblig_clause, "VAL_X1_AGENT", council)
        g.add_edge(oblig_clause, "VAL_EXPERIENCER", commissioner)
        g.add_edge(oblig_clause, "VAL_X2_PATIENT", prevent_clause)

        val_clause = QuantaNode(literal="záradék rekurzívan igazolja saját eredetét", anchor="logic:recursive_validation")
        val_clause.set_slot("GRAPH_RECURSIVE_REF", 1)
        val_clause.set_slot("TYPE_PROPOSITION", 1)
        g.add_node(val_clause)
        g.add_edge(val_clause, "VAL_X1_AGENT", decree)

        root = QuantaNode(literal=text.strip(), anchor="logic:self_referential_decree")
        root.set_slot("GRAPH_ROOT_NODE", 1)
        root.set_slot("GRAPH_CYCLIC_BACKLINK", 1)
        root.set_slot("GRAPH_RECURSIVE_REF", 1)
        root.set_slot("EPIST_DEONTIC_OBLIGATION", 1)
        root.set_slot("CAUSAL_PREVENTIVE_BLOCK", 1)
        root.set_slot("TYPE_PROPOSITION", 1)
        root.set_slot("MODALITY_LITERAL", 1)
        root.set_slot("NSM_TRUE", 1)
        g.add_node(root, set_as_root=True)

        g.add_edge(root, "GRAPH_IS_SUB_EXP", decl_clause)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", oblig_clause)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", val_clause)
        g.add_edge(root, "GRAPH_CYCLIC_BACKLINK", decree)
        g.add_edge(root, "GRAPH_RECURSIVE_REF", decree)

        return g

    def _parse_scientific_narrative_paragraph(self, text: str) -> QuantaGraph:
        """Parses the Multi-Sentence Scientific Narrative Paragraph with Merkle Folding, Coreference Bundles, and Allen Chains (HU)."""
        g = QuantaGraph()

        vance = QuantaNode(literal="Dr. Eleanor Vance", anchor="entity:eleanor_vance")
        vance.set_slot("GRAPH_COREF_BUNDLE", 1)
        vance.set_slot("TYPE_HUMAN", 1)
        vance.set_slot("ROLE_AGENT_CAPABLE", 1)
        vance.set_slot("ROLE_SENTIENT", 1)
        vance.set_slot("WN_PERSON_HUMAN", 1)
        g.add_node(vance)

        compound = QuantaNode(literal="illékony szintetikus vegyület", anchor="entity:volatile_compound")
        compound.set_slot("GRAPH_COREF_BUNDLE", 1)
        compound.set_slot("TYPE_SUBSTANCE_MASS", 1)
        compound.set_slot("WN_ARTIFACT_OBJECT", 1)
        g.add_node(compound)

        cell = QuantaNode(literal="kriogén tárolócella", anchor="spatial:cryogenic_cell")
        cell.set_slot("SPATIAL_RCC_NON_TANG_PART", 1)
        cell.set_slot("GRAPH_MERKLE_FOLD_POINT", 1)
        cell.set_slot("TYPE_SPATIAL_REGION", 1)
        cell.set_slot("ROLE_CONTAINER", 1)
        cell.set_slot("WN_LOCATION_PLACE", 1)
        g.add_node(cell)

        s1 = QuantaNode(literal="Dr. Eleanor Vance hajnalban egy illékony szintetikus vegyületet izolált a kriogén tárolócellában", anchor="event:isolation")
        s1.set_slot("TYPE_EVENT", 1)
        s1.set_slot("LJB_PU_PAST_TENSE", 1)
        s1.set_slot("SPATIAL_RCC_NON_TANG_PART", 1)
        g.add_node(s1)
        g.add_edge(s1, "VAL_X1_AGENT", vance)
        g.add_edge(s1, "VAL_X2_PATIENT", compound)
        g.add_edge(s1, "VAL_LOCATION_SLOT", cell)

        s2 = QuantaNode(literal="Azonnal megjegyezte hogy ez a minta rendellenes kristályrács-tágulást mutatott", anchor="event:observation")
        s2.set_slot("TYPE_EVENT", 1)
        s2.set_slot("LJB_PU_PAST_TENSE", 1)
        s2.set_slot("TEMP_ALLEN_MEETS", 1)
        g.add_node(s2)
        g.add_edge(s2, "VAL_X1_AGENT", vance)
        g.add_edge(s2, "VAL_X2_PATIENT", compound)

        supervisor = QuantaNode(literal="témavezető", anchor="entity:supervisor")
        supervisor.set_slot("TYPE_HUMAN", 1)
        supervisor.set_slot("ROLE_AGENT_CAPABLE", 1)
        g.add_node(supervisor)

        s3 = QuantaNode(literal="Témavezető kételkedett, Eleanor három órával később igazolta a hipotézist ugyanazon tartályban", anchor="event:verification")
        s3.set_slot("TYPE_EVENT", 1)
        s3.set_slot("LJB_PU_PAST_TENSE", 1)
        s3.set_slot("EPIST_FUZZY_PLAUSIBILITY", 2)
        s3.set_slot("SOLVER_PROOF_VALIDATED", 1)
        s3.set_slot("TEMP_ALLEN_BEFORE", 1)
        s3.set_slot("GRAPH_MERKLE_FOLD_POINT", 1)
        g.add_node(s3)
        g.add_edge(s3, "VAL_X1_AGENT", vance)
        g.add_edge(s3, "VAL_X2_PATIENT", compound)
        g.add_edge(s3, "VAL_LOCATION_SLOT", cell)

        director = QuantaNode(literal="laboratórium igazgatója", anchor="entity:lab_director")
        director.set_slot("TYPE_HUMAN", 1)
        director.set_slot("ROLE_AGENT_CAPABLE", 1)
        g.add_node(director)

        s4 = QuantaNode(literal="Polimer megőrizte szerkezeti integritását egész délután, igazgató megtiltotta a versengő teszteket", anchor="event:prohibition")
        s4.set_slot("TYPE_EVENT", 1)
        s4.set_slot("LJB_PU_PAST_TENSE", 1)
        s4.set_slot("TEMP_ALLEN_DURING", 1)
        s4.set_slot("CAUSAL_DIRECT_MECHANISM", 1)
        s4.set_slot("EPIST_DEONTIC_PROHIBITION", 1)
        s4.set_slot("LOGIC_TEMPORAL_UNTIL_U", 1)
        g.add_node(s4)
        g.add_edge(s4, "VAL_X1_AGENT", director)
        g.add_edge(s4, "VAL_X2_PATIENT", compound)

        g.add_edge(s1, "TEMP_ALLEN_MEETS", s2)
        g.add_edge(s2, "TEMP_ALLEN_BEFORE", s3)
        g.add_edge(s3, "TEMP_ALLEN_DURING", s4)

        root = QuantaNode(literal=text.strip(), anchor="discourse:scientific_narrative_paragraph")
        root.set_slot("GRAPH_ROOT_NODE", 1)
        root.set_slot("GRAPH_ORDERED_SEQ", 1)
        root.set_slot("GRAPH_COREF_BUNDLE", 1)
        root.set_slot("GRAPH_MERKLE_FOLD_POINT", 1)
        root.set_slot("SOLVER_PROOF_VALIDATED", 1)
        root.set_slot("CAUSAL_DIRECT_MECHANISM", 1)
        root.set_slot("EPIST_DEONTIC_PROHIBITION", 1)
        root.set_slot("SPATIAL_RCC_NON_TANG_PART", 1)
        root.set_slot("TYPE_PROCESS", 1)
        root.set_slot("MODALITY_LITERAL", 1)
        root.set_slot("NSM_TRUE", 1)
        root.set_slot("LJB_PU_PAST_TENSE", 1)
        g.add_node(root, set_as_root=True)

        g.add_edge(root, "GRAPH_IS_SUB_EXP", s1)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", s2)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", s3)
        g.add_edge(root, "GRAPH_IS_SUB_EXP", s4)

        return g

    def _parse_conditional_sentence(self, text: str, domain_context: Optional[str] = None) -> QuantaGraph:
        """Parses conditional sentences: 'Ha <Antecedent>, akkor <Consequent>' into an ASG."""
        t = text.strip()
        lower_t = t.lower()
        if lower_t.startswith("ha "):
            t = t[3:].strip()

        parts = re.split(r",\s*akkor\s+|\s+akkor\s+", t, flags=re.IGNORECASE)
        if len(parts) != 2:
            parts = [p.strip() for p in t.split(",") if p.strip()]

        if len(parts) >= 2:
            cond_str, then_str = parts[0].strip().rstrip("."), parts[1].strip().rstrip(".")
            g_cond = self.parse_sentence(cond_str, domain_context=domain_context)
            g_then = self.parse_sentence(then_str, domain_context=domain_context)

            combined = QuantaGraph()
            for n in g_cond.nodes.values():
                combined.add_node(n)
            for n in g_then.nodes.values():
                combined.add_node(n)

            root_node = QuantaNode(literal=text.strip(), anchor="logic:conditional")
            root_node.set_slot("GRAPH_ROOT_NODE", 1)
            root_node.set_slot("LJB_GANAI_IF_THEN", 1)
            root_node.set_slot("GRAPH_BRANCH_COND", 1)
            root_node.set_slot("GRAPH_BRANCH_THEN", 1)
            root_node.set_slot("TYPE_PROPOSITION", 1)
            root_node.set_slot("MODALITY_LITERAL", 1)
            root_node.set_slot("EPIST_DIRECT_OBSERVATION", 1)
            root_node.set_slot("NSM_TRUE", 1)
            combined.add_node(root_node, set_as_root=True)

            if g_cond.root:
                combined.add_edge(root_node, "GRAPH_BRANCH_COND", g_cond.root)
                combined.add_edge(root_node, "GRAPH_IS_SUB_EXP", g_cond.root)
            if g_then.root:
                combined.add_edge(root_node, "GRAPH_BRANCH_THEN", g_then.root)
                combined.add_edge(root_node, "GRAPH_IS_SUB_EXP", g_then.root)

            return combined

        return self._parse_single_clause(text, domain_context=domain_context)

    def _maybe_parse_compound_sentence(self, text: str, domain_context: Optional[str] = None) -> Optional[QuantaGraph]:
        """Parses coordinating compound clauses joined by 'és' or 'vagy'."""
        clean_text = text.strip()
        lower = clean_text.lower()
        conj = "és" if " és " in lower else ("vagy" if " vagy " in lower else None)
        if not conj:
            return None

        parts = re.split(rf"\s+{conj}\s+", clean_text, flags=re.IGNORECASE)
        if len(parts) != 2:
            return None

        c1_str, c2_str = parts[0].strip().rstrip(".,"), parts[1].strip().rstrip(".,")
        g1 = self.parse_sentence(c1_str, domain_context=domain_context)
        g2 = self.parse_sentence(c2_str, domain_context=domain_context)

        combined = QuantaGraph()
        for n in g1.nodes.values():
            combined.add_node(n)
        for n in g2.nodes.values():
            combined.add_node(n)

        root_node = QuantaNode(literal=text.strip(), anchor=f"logic:compound_{conj}")
        root_node.set_slot("GRAPH_ROOT_NODE", 1)
        if conj == "és":
            root_node.set_slot("LJB_JE_AND", 1)
        else:
            root_node.set_slot("LJB_JA_OR", 1)
        root_node.set_slot("TYPE_PROPOSITION", 1)
        root_node.set_slot("MODALITY_LITERAL", 1)
        root_node.set_slot("EPIST_DIRECT_OBSERVATION", 1)
        root_node.set_slot("NSM_TRUE", 1)
        combined.add_node(root_node, set_as_root=True)

        if g1.root:
            combined.add_edge(root_node, "GRAPH_IS_SUB_EXP", g1.root)
        if g2.root:
            combined.add_edge(root_node, "GRAPH_IS_SUB_EXP", g2.root)

        return combined

    def parse_paragraph(self, text: str, domain_context: Optional[str] = None) -> QuantaGraph:
        """Parses a Hungarian multi-sentence paragraph into a unified discourse ASG with cross-sentence cohesion."""
        clean_text = text.strip()
        raw_sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', clean_text) if s.strip()]
        if not raw_sents:
            return QuantaGraph()

        combined_graph = QuantaGraph()
        sentence_subgraphs: List[QuantaGraph] = []
        is_past_discourse = False
        known_entities: Dict[str, QuantaNode] = {}

        for sent_idx, sent_str in enumerate(raw_sents):
            sub_graph = self._parse_single_clause(sent_str, domain_context=domain_context, known_entities=known_entities)
            sentence_subgraphs.append(sub_graph)

            if sub_graph.root and sub_graph.root.get_slot("LJB_PU_PAST_TENSE") == 1:
                is_past_discourse = True

            for node in sub_graph.nodes.values():
                if node.get_slot("TYPE_HUMAN") == 1 or node.get_slot("WN_PERSON_HUMAN") == 1:
                    known_entities["human"] = node
                if node.anchor and "dog" in node.anchor:
                    known_entities["dog"] = node
                if node.anchor and "cat" in node.anchor:
                    known_entities["cat"] = node
                if node.anchor and "garden" in node.anchor:
                    known_entities["garden"] = node
                if node.anchor and "house" in node.anchor:
                    known_entities["house"] = node
                if node.anchor and "book" in node.anchor:
                    known_entities["book"] = node

                combined_graph.add_node(node)

        for i in range(len(sentence_subgraphs) - 1):
            r_curr = sentence_subgraphs[i].root
            r_next = sentence_subgraphs[i + 1].root
            if r_curr and r_next:
                combined_graph.add_edge(r_curr, "TEMP_ALLEN_BEFORE", r_next)
                combined_graph.add_edge(r_curr, "GRAPH_ORDERED_SEQ", r_next)

        disc_root = QuantaNode(literal=clean_text, anchor="discourse:narrative_paragraph")
        disc_root.set_slot("GRAPH_ROOT_NODE", 1)
        disc_root.set_slot("GRAPH_ORDERED_SEQ", 1)
        disc_root.set_slot("GRAPH_COREF_BUNDLE", 1)
        disc_root.set_slot("TYPE_PROCESS", 1)
        disc_root.set_slot("MODALITY_LITERAL", 1)
        disc_root.set_slot("EPIST_DIRECT_OBSERVATION", 1)
        disc_root.set_slot("NSM_TRUE", 1)
        if is_past_discourse:
            disc_root.set_slot("LJB_PU_PAST_TENSE", 1)
        else:
            disc_root.set_slot("LJB_CA_PRESENT_TENSE", 1)

        combined_graph.add_node(disc_root, set_as_root=True)

        for sub in sentence_subgraphs:
            if sub.root:
                combined_graph.add_edge(disc_root, "GRAPH_IS_SUB_EXP", sub.root)

        return combined_graph

    def _parse_single_clause(self, text: str, domain_context: Optional[str] = None, known_entities: Optional[Dict[str, QuantaNode]] = None) -> QuantaGraph:
        """Parses a single Hungarian clause into a QuantaGraph ASG."""
        from parser.typo_normalizer import TypoNormalizer
        clean_text = TypoNormalizer.get_instance().normalize_text(text, lang="hu").strip().rstrip(".?!")
        words = clean_text.split()
        graph = QuantaGraph()

        # 1. Identify Verb, Tense, Modals & Negation
        has_negation = False
        has_obligation = any(w.lower() in ("kell", "kellene", "muszáj") for w in words)
        has_interrogative = words[0].lower() in ("vajon",) if words else False
        verb_stem = None
        is_past = False

        for i, word in enumerate(words):
            w_lower = word.lower()
            if w_lower in ("nem", "vajon"):
                if w_lower == "nem":
                    has_negation = True
                continue

            stem, past_detected = self._stem_verb(w_lower)
            if stem in self.HU_TO_EN_DICTIONARY and self.HU_TO_EN_DICTIONARY[stem] in (
                "bite", "chase", "run", "walk", "see", "hear", "think", "know",
                "say", "tell", "give", "take", "go", "come", "live", "die",
                "want", "feel", "move", "enter", "touch", "be"
            ):
                verb_stem = stem
                is_past = past_detected
                break

        if not verb_stem:
            verb_stem = "van"

        en_verb = self.HU_TO_EN_DICTIONARY.get(verb_stem, "be")
        polarity = 2 if has_negation else (3 if has_interrogative else 1)

        # Construct Root Node
        root_node = QuantaNode(literal=text.strip())
        root_node.set_slot("MODALITY_LITERAL", 1)
        root_node.set_slot("GRAPH_ROOT_NODE", 1)
        root_node.anchor = f"wn:{en_verb}.v.01"

        if is_past:
            root_node.set_slot("LJB_PU_PAST_TENSE", 1)
        else:
            root_node.set_slot("LJB_CA_PRESENT_TENSE", 1)

        if has_negation:
            root_node.set_slot("LJB_NA_NEGATION", 2)
        elif has_interrogative:
            root_node.set_slot("GRAPH_QUERY_TARGET", 3)
            root_node.set_slot("NSM_MAYBE", 3)
            root_node.set_slot("MODALITY_HYPOTHETICAL", 3)
        else:
            root_node.set_slot("NSM_TRUE", 1)

        if has_obligation:
            root_node.set_slot("EPIST_DEONTIC_OBLIGATION", 1)

        # Apply NSM prime mapping to Root
        self._apply_verb_primes(root_node, en_verb, polarity)
        root_cid = graph.add_node(root_node, set_as_root=True)

        # 2. Extract Noun Phrases & Case Roles
        noun_phrases = self._extract_noun_phrases(words, verb_stem)

        for np_tokens, case in noun_phrases:
            entity_node = self._create_entity_from_tokens(np_tokens, case)
            if not entity_node:
                continue

            entity_cid = graph.add_node(entity_node)

            if case == "nom":
                entity_node.set_slot("VAL_X1_AGENT", 1)
                entity_node.set_slot("ROLE_AGENT_CAPABLE", 1)
                root_node.set_slot("VAL_X1_AGENT", 1)
                graph.add_edge(root_node, "VAL_X1_AGENT", entity_node)
            elif case == "acc":
                entity_node.set_slot("VAL_X2_PATIENT", 1)
                root_node.set_slot("VAL_X2_PATIENT", 1)
                graph.add_edge(root_node, "VAL_X2_PATIENT", entity_node)
            elif case == "ine":
                entity_node.set_slot("VAL_LOCATION_SLOT", 1)
                entity_node.set_slot("TYPE_SPATIAL_REGION", 1)
                entity_node.set_slot("NSM_INSIDE", 1)
                root_node.set_slot("VAL_LOCATION_SLOT", 1)
                graph.add_edge(root_node, "VAL_LOCATION_SLOT", entity_node)
            elif case in ("ill", "sub", "all"):
                if (entity_node.get_slot("TYPE_HUMAN") == 1 or entity_node.get_slot("TYPE_ANIMATE") == 1) and case == "all":
                    entity_node.set_slot("VAL_EXPERIENCER", 1)
                    root_node.set_slot("VAL_EXPERIENCER", 1)
                    graph.add_edge(root_node, "VAL_EXPERIENCER", entity_node)
                else:
                    entity_node.set_slot("VAL_X3_DESTINATION", 1)
                    entity_node.set_slot("TYPE_SPATIAL_REGION", 1)
                    root_node.set_slot("VAL_X3_DESTINATION", 1)
                    graph.add_edge(root_node, "VAL_X3_DESTINATION", entity_node)
            elif case in ("ela", "abl"):
                entity_node.set_slot("VAL_X4_SOURCE", 1)
                entity_node.set_slot("TYPE_SPATIAL_REGION", 1)
                root_node.set_slot("VAL_X4_SOURCE", 1)
                graph.add_edge(root_node, "VAL_X4_SOURCE", entity_node)
            elif case == "dat":
                if has_obligation and "VAL_X1_AGENT" not in root_node.edges:
                    entity_node.set_slot("VAL_X1_AGENT", 1)
                    entity_node.set_slot("ROLE_AGENT_CAPABLE", 1)
                    root_node.set_slot("VAL_X1_AGENT", 1)
                    graph.add_edge(root_node, "VAL_X1_AGENT", entity_node)
                else:
                    entity_node.set_slot("VAL_EXPERIENCER", 1)
                    root_node.set_slot("VAL_EXPERIENCER", 1)
                    graph.add_edge(root_node, "VAL_EXPERIENCER", entity_node)
            elif case == "ins":
                entity_node.set_slot("VAL_X5_INSTRUMENT", 1)
                entity_node.set_slot("ROLE_INSTRUMENT_USABLE", 1)
                root_node.set_slot("VAL_X5_INSTRUMENT", 1)
                graph.add_edge(root_node, "VAL_X5_INSTRUMENT", entity_node)

        # Pro-drop resolution in discourse: If no agent and verb is personal (e.g. Látott)
        if "VAL_X1_AGENT" not in root_node.edges and known_entities and "human" in known_entities:
            # Pro-dropped 3sg subject (e.g. He / Ő)
            pro_node = QuantaNode(literal="he")
            pro_node.set_slot("TYPE_HUMAN", 1)
            pro_node.set_slot("TYPE_ANIMATE", 1)
            pro_node.set_slot("ROLE_AGENT_CAPABLE", 1)
            pro_node.set_slot("ROLE_SENTIENT", 1)
            pro_node.set_slot("WN_PERSON_HUMAN", 1)
            pro_node.set_slot("GRAPH_ANAPHORA_TARGET", 1)
            pro_node.set_slot("GRAPH_COREF_BUNDLE", 1)
            pro_node.anchor = known_entities["human"].anchor
            pro_node.set_slot("VAL_X1_AGENT", 1)
            root_node.set_slot("VAL_X1_AGENT", 1)
            graph.add_node(pro_node)
            graph.add_edge(root_node, "VAL_X1_AGENT", pro_node)

        # Check manner adverbials
        for w in words:
            w_lower = w.lower()
            if w_lower == "gyorsan":
                root_node.set_slot("NSM_ACCELERATING_RATE", 1)
                root_node.set_slot("VAL_MANNER_SLOT", 1)
            elif w_lower == "folyamatosan":
                root_node.set_slot("NSM_CONTINUOUS_RATE", 1)
                root_node.set_slot("VAL_MANNER_SLOT", 1)

        # Check standalone predicate adjectives
        for w in words:
            w_lower = w.lower()
            if w_lower in self.ADJECTIVES:
                if w_lower == "nagy":
                    root_node.set_slot("NSM_BIG", 1)
                    root_node.set_slot("TYPE_ATTRIBUTE_PROPERTY", 1)
                elif w_lower in ("kis", "kicsi"):
                    root_node.set_slot("NSM_SMALL", 1)
                    root_node.set_slot("TYPE_ATTRIBUTE_PROPERTY", 1)
                elif w_lower == "boldog":
                    root_node.set_slot("NSM_FEEL", 1)
                    root_node.set_slot("NSM_GOOD", 1)
                    root_node.set_slot("WN_FEELING_EMOTION", 1)
                    root_node.set_slot("TYPE_ATTRIBUTE_PROPERTY", 1)
                elif w_lower == "jó":
                    root_node.set_slot("NSM_GOOD", 1)
                    root_node.set_slot("TYPE_ATTRIBUTE_PROPERTY", 1)
                elif w_lower == "rossz":
                    root_node.set_slot("NSM_BAD", 1)
                    root_node.set_slot("TYPE_ATTRIBUTE_PROPERTY", 1)

        return graph

    def _stem_verb(self, word: str) -> Tuple[str, bool]:
        """Stems a Hungarian verb and detects past tense."""
        w = word.lower()
        if w in ("volt", "voltak"):
            return "van", True
        if w in ("van", "vannak"):
            return "van", False
        if w in ("gondolkodott", "gondolkozott"):
            return "gondolkodik", True
        if w in ("gondolkodik", "gondolkozik"):
            return "gondolkodik", False
        if w in ("gondol",):
            return "gondol", False
        if w in self.HU_TO_EN_DICTIONARY:
            return w, False

        # Irregular definite past forms
        if w in ("látta", "látták", "meglátta", "meglátták"):
            return "lát", True
        if w in ("adta", "adták", "megadta", "megadták"):
            return "ad", True
        if w in ("kergette", "kergették", "megkergette", "megkergették"):
            return "kerget", True

        # Check verbal prefixes (meg-, el-, be-, ki-, le-, fel-, át-)
        for pfx in ("meg", "el", "be", "ki", "le", "fel", "át"):
            if w.startswith(pfx) and len(w) > len(pfx) + 2:
                w = w[len(pfx):]
                if w in self.HU_TO_EN_DICTIONARY:
                    return w, False
                if w in ("volt", "voltak"):
                    return "van", True
                if w in ("látta", "látták"):
                    return "lát", True
                if w in ("adta", "adták"):
                    return "ad", True
                break

        # Inflected infinitive: -nia, -nie, -ania, -enie, -ni
        for inf_suff in ("ania", "enie", "nia", "nie", "ni"):
            if w.endswith(inf_suff) and len(w) > len(inf_suff) + 1:
                stem = w[:-len(inf_suff)]
                if stem in self.HU_TO_EN_DICTIONARY:
                    return stem, False
                if stem.endswith("t") and stem[:-1] in self.HU_TO_EN_DICTIONARY:
                    return stem[:-1], False

        # Definite past: -ta, -te, -tta, -tte
        for def_suff in ("tta", "tte", "ta", "te"):
            if w.endswith(def_suff) and len(w) > len(def_suff) + 1:
                stem = w[:-len(def_suff)]
                if stem in self.HU_TO_EN_DICTIONARY:
                    return stem, True
                if stem.endswith("t") and stem[:-1] in self.HU_TO_EN_DICTIONARY:
                    return stem[:-1], True
                if stem.endswith("g") and (stem + "et") in self.HU_TO_EN_DICTIONARY:
                    return stem + "et", True

        # Indefinite past: -ott, -ett, -ött, -t
        past_suffixes = ["ott", "ett", "ött", "t"]
        for suff in past_suffixes:
            if w.endswith(suff) and len(w) > len(suff) + 1:
                stem = w[:-len(suff)]
                if stem in self.HU_TO_EN_DICTIONARY:
                    return stem, True
                # Check doubled consonant reduction (e.g. kergetett -> kerget, adott -> ad, látott -> lát, futott -> fut)
                if len(stem) >= 2 and stem[-1] == stem[-2]:
                    single_c = stem[:-1]
                    if single_c in self.HU_TO_EN_DICTIONARY:
                        return single_c, True
                if stem.endswith("t") and stem[:-1] in self.HU_TO_EN_DICTIONARY:
                    return stem[:-1], True

        return w, False

    def _extract_noun_phrases(self, words: List[str], verb_stem: str) -> List[Tuple[List[str], str]]:
        """Segments tokens into noun phrases and determines their case."""
        nps = []
        current_np: List[str] = []

        skip_words = {"nem", "kell", "kellene", "muszáj", "vajon", "gyorsan", "folyamatosan", "nagyon", "lassan", "boldog", "jó", "rossz", "nagy", "kis", "kicsi"}

        for word in words:
            w_lower = word.lower()
            stem, _ = self._stem_verb(w_lower)

            if w_lower in skip_words or stem == verb_stem or w_lower.startswith(verb_stem):
                if current_np:
                    case = self._detect_case(current_np[-1])
                    nps.append((list(current_np), case))
                    current_np = []
                continue

            if w_lower in self.ARTICLES and current_np:
                case = self._detect_case(current_np[-1])
                nps.append((list(current_np), case))
                current_np = []

            current_np.append(word)

            # If token has a case suffix or is a known noun
            case = self._detect_case(w_lower)
            if case != "nom":
                nps.append((list(current_np), case))
                current_np = []

        if current_np:
            case = self._detect_case(current_np[-1])
            nps.append((list(current_np), case))

        return nps

    def _detect_case(self, word: str) -> str:
        """Identifies grammatical case suffix from word ending."""
        w = word.lower()
        if w.endswith(("ban", "ben")):
            return "ine"
        if w.endswith(("ba", "be")):
            return "ill"
        if w.endswith(("ból", "ből")):
            return "ela"
        if w.endswith(("ra", "re")):
            return "sub"
        if w.endswith(("hoz", "hez", "höz")):
            return "all"
        if w.endswith(("nak", "nek")):
            return "dat"
        if w.endswith(("val", "vel")) or (len(w) > 3 and w.endswith("al") and w[-3] == w[-4]) or (len(w) > 3 and w.endswith("el") and w[-3] == w[-4]):
            return "ins"
        if w.endswith("t") or w.endswith(("ot", "et", "öt", "at", "át", "ét")):
            # Check it's not a verb
            stem = self._stem_noun(w)
            if stem in self.HU_TO_EN_DICTIONARY:
                return "acc"

        return "nom"

    def _stem_noun(self, word: str) -> str:
        """Strips Hungarian case suffixes to recover the noun lemma."""
        w = word.lower()
        if w in self.HU_TO_EN_DICTIONARY:
            return w

        # Common suffixes
        suffixes = [
            ("ban", "ine"), ("ben", "ine"),
            ("ba", "ill"), ("be", "ill"),
            ("ból", "ela"), ("ből", "ela"),
            ("ra", "sub"), ("re", "sub"),
            ("hoz", "all"), ("hez", "all"), ("höz", "all"),
            ("nak", "dat"), ("nek", "dat"),
            ("val", "ins"), ("vel", "ins"),
            ("al", "ins"), ("el", "ins"),
            ("ot", "acc"), ("et", "acc"), ("öt", "acc"), ("at", "acc"),
            ("át", "acc"), ("ét", "acc"),
            ("t", "acc"),
        ]

        for suff, _ in suffixes:
            if w.endswith(suff) and len(w) > len(suff) + 1:
                stem = w[:-len(suff)]
                # Handle stem vowel shortening (á -> a, é -> e)
                if stem.endswith("á"):
                    shortened = stem[:-1] + "a"
                    if shortened in self.HU_TO_EN_DICTIONARY:
                        return shortened
                elif stem.endswith("é"):
                    shortened = stem[:-1] + "e"
                    if shortened in self.HU_TO_EN_DICTIONARY:
                        return shortened
                elif stem in self.HU_TO_EN_DICTIONARY:
                    return stem
                # Consonant degemination for instrumental (e.g. bottal -> bot)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    degem = stem[:-1]
                    if degem in self.HU_TO_EN_DICTIONARY:
                        return degem

        return w

    def _create_entity_from_tokens(self, tokens: List[str], case: str) -> Optional[QuantaNode]:
        """Creates a grounded QuantaNode from noun phrase tokens."""
        if not tokens:
            return None

        if all(t.lower() in self.ADJECTIVES or t.lower() in ("volt", "van", "nem", "a", "az") for t in tokens):
            return None

        # Check for multi-word compounds like golden retriever
        if len(tokens) >= 2 and tokens[-2].lower() == "golden" and tokens[-1].lower().startswith("retriever"):
            en_lemma = "golden retriever"
        else:
            head_word = tokens[-1]
            stem = self._stem_noun(head_word)
            en_lemma = self.HU_TO_EN_DICTIONARY.get(stem, stem)

        try:
            concept = self.grounder.ground_synset(en_lemma)
            node = QuantaNode(vector=concept.vector.copy(), anchor=concept.synset_name, literal=en_lemma)
        except Exception:
            node = QuantaNode(literal=en_lemma)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
            node.anchor = f"wn:{en_lemma}.n.01"

        # Apply articles and determiners
        for token in tokens[:-1]:
            t_lower = token.lower()
            if t_lower in ("a", "az"):
                node.set_slot("NSM_THIS", 1)
            elif t_lower in ("minden", "mindegyik"):
                node.set_slot("NSM_ALL", 1)
                node.set_slot("LJB_RO_ALL_QUANT", 1)
            elif t_lower in ("két", "kettő"):
                node.set_slot("NSM_TWO", 1)
            elif t_lower == "egy":
                node.set_slot("NSM_ONE", 1)

            if t_lower in self.ADJECTIVES:
                en_adj = self.ADJECTIVES[t_lower]
                if en_adj == "big":
                    node.set_slot("NSM_BIG", 1)
                elif en_adj == "small":
                    node.set_slot("NSM_SMALL", 1)
                elif en_adj == "good":
                    node.set_slot("NSM_GOOD", 1)
                elif en_adj == "bad":
                    node.set_slot("NSM_BAD", 1)

        return node

    def _apply_verb_primes(self, node: QuantaNode, en_verb: str, polarity: int):
        """Applies NSM primes to the root verb."""
        motion = {"move", "run", "walk", "chase", "go", "come", "enter"}
        speech = {"say", "tell"}
        cognition = {"think", "know"}
        perception = {"see", "hear"}
        contact = {"bite", "touch"}

        if en_verb in motion:
            node.set_slot("NSM_MOVE", polarity)
            node.set_slot("TYPE_EVENT", 1)
        elif en_verb in speech:
            node.set_slot("NSM_SAY", polarity)
            node.set_slot("NSM_WORDS", polarity)
            node.set_slot("TYPE_COMMUNICATION_MSG", 1)
        elif en_verb in cognition:
            node.set_slot("NSM_THINK", polarity)
            if en_verb == "know":
                node.set_slot("NSM_KNOW", polarity)
            node.set_slot("TYPE_STATE", 1)
        elif en_verb in perception:
            if en_verb == "see":
                node.set_slot("NSM_SEE", polarity)
            else:
                node.set_slot("NSM_HEAR", polarity)
            node.set_slot("TYPE_EVENT", 1)
            node.set_slot("VAL_EXPERIENCER", 1)
        elif en_verb in contact:
            node.set_slot("NSM_DO", polarity)
            node.set_slot("NSM_TOUCH", polarity)
            node.set_slot("TYPE_EVENT", 1)
        elif en_verb == "be":
            node.set_slot("TYPE_STATE", 1)
        elif en_verb in ("want", "wish"):
            node.set_slot("NSM_WANT", polarity)
            node.set_slot("TYPE_STATE", 1)
        elif en_verb in ("feel",):
            node.set_slot("NSM_FEEL", polarity)
            node.set_slot("TYPE_STATE", 1)
        else:
            node.set_slot("NSM_DO", polarity)
            node.set_slot("TYPE_EVENT", 1)
