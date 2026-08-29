"""Natural Language Forward Parser for QUANTA.

Translates English sentences into Content-Addressed Abstract Syntax Graphs (ASGs)
and 256-dimensional quaternary semantic vectors.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import spacy

try:
    from nltk.corpus import wordnet as wn
    NLTK_WN = True
except ImportError:
    NLTK_WN = False

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_name
from core.types import QuantaVector, QuaternaryValue
from parser.lexical_grounder import WordNetLexicalGrounder


class SVOResult(NamedTuple):
    """Container for Subject-Verb-Object and modifiers extracted from dependency tree."""
    subject: Optional[str]
    verb: Optional[str]
    object: Optional[str]
    modifiers: List[Dict[str, Any]]


class NLPForwardParser:
    """End-to-end forward parser mapping natural language text to Quanta ASG topologies."""

    def __init__(self, spacy_model: str = "en_core_web_sm", offline_cache_path: Optional[str] = None):
        self.nlp = spacy.load(spacy_model)
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)

        # Lexical classification maps for Band 0 NSM primes
        self.motion_verbs = {
            "move", "go", "walk", "run", "travel", "fly", "jump", "enter", "leave",
            "cross", "drive", "ride", "step", "pass", "shift", "head", "crawl", "swim",
            "chase", "pursue", "follow", "arrive", "depart", "reach", "approach", "climb",
            "wander", "rush", "journey", "flee", "escape", "return", "navigate",
        }
        self.speech_verbs = {
            "say", "tell", "speak", "talk", "utter", "claim", "declare", "state",
            "ask", "reply", "answer", "announce", "shout", "whisper", "report", "describe",
            "explain", "argue", "mention", "assert", "remark", "suggest", "command", "warn",
        }
        self.cognition_verbs = {
            "think", "believe", "know", "understand", "remember", "forget", "realize",
            "suppose", "assume", "conclude", "infer", "deduce", "judge", "consider",
            "reckon", "reflect", "ponder", "comprehend", "recognize",
        }
        self.volition_verbs = {
            "want", "desire", "wish", "need", "prefer", "crave", "hope", "plan", "intend",
            "aspire", "aim", "seek", "long", "choose", "like", "love",
        }
        self.perception_verbs = {
            "see", "look", "watch", "observe", "perceive", "notice", "view", "glance", "spot",
        }
        self.auditory_verbs = {
            "hear", "listen", "sound", "overhear",
        }
        self.contact_verbs = {
            "touch", "hit", "strike", "bite", "grab", "hold", "catch", "push", "pull",
            "take", "drop", "kick", "carry", "place", "put", "grasp", "seize", "press",
            "rub", "scratch", "hug", "kiss", "pat", "tap", "slam",
        }
        self.possession_verbs = {
            "have", "own", "possess", "hold", "keep", "acquire", "get", "give", "receive",
            "buy", "sell", "borrow", "lend", "belong", "obtain", "gain",
        }
        self.life_verbs = {
            "live", "die", "born", "grow", "breathe", "perish", "survive", "decay", "sprout",
        }
        self.event_verbs = {
            "happen", "occur", "transpire", "arise", "unfold", "materialize",
        }

    def parse_dependency_tree(self, text: str) -> Any:
        """Parses text into a spaCy Doc dependency tree after typo normalization."""
        from parser.typo_normalizer import TypoNormalizer
        clean_text = TypoNormalizer.get_instance().normalize_text(text, lang="en")
        return self.nlp(clean_text.strip())

    def extract_subject_verb_object(self, doc: Any) -> SVOResult:
        """Extracts (subject, verb, object, modifiers) from a spaCy dependency tree."""
        known_verb_lemmas = {
            "bit": "bite", "bites": "bite", "bite": "bite",
            "chased": "chase", "chases": "chase", "chase": "chase",
            "ran": "run", "runs": "run", "run": "run",
            "saw": "see", "sees": "see", "see": "see",
            "gave": "give", "gives": "give", "give": "give",
            "walked": "walk", "walks": "walk", "walk": "walk",
            "thought": "think", "thinks": "think", "think": "think",
            "knew": "know", "knows": "know", "know": "know",
            "wanted": "want", "wants": "want", "want": "want",
            "felt": "feel", "feels": "feel", "feel": "feel",
            "touched": "touch", "touches": "touch", "touch": "touch",
        }

        root_token = None
        for token in doc:
            if token.dep_ == "ROOT":
                root_token = token
                break

        if root_token is not None and root_token.pos_ != "VERB":
            for token in doc:
                if token.text.lower() in known_verb_lemmas or token.pos_ == "VERB":
                    root_token = token
                    break

        if root_token is None and len(doc) > 0:
            root_token = doc[0]

        # Extract Subject
        agent_token = None
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass", "csubj") and (token.head == root_token or token.i < root_token.i):
                agent_token = token
                break

        if agent_token is None and root_token:
            for token in doc:
                if token.i < root_token.i and token.pos_ in ("NOUN", "PROPN", "PRON") and token.dep_ not in ("prep", "pobj", "det"):
                    agent_token = token
                    break

        subject_str = None
        if agent_token:
            compounds = [c for c in agent_token.children if c.dep_ in ("compound", "amod") and c.i < agent_token.i]
            if not compounds:
                prev_tokens = [doc[i] for i in range(max(0, agent_token.i - 2), agent_token.i) if doc[i].pos_ in ("ADJ", "NOUN") and doc[i].dep_ not in ("det", "prep")]
                compounds = prev_tokens
            if compounds:
                subject_str = " ".join([c.text for c in compounds] + [agent_token.text])
            else:
                subject_str = agent_token.text

        # Extract Object
        patient_token = None
        for token in doc:
            if token.dep_ in ("dobj", "attr", "dative", "acomp", "oprd") and (token.head == root_token or token.i > root_token.i):
                patient_token = token
                break

        if patient_token is None and root_token:
            for token in doc:
                if token.i > root_token.i and token.dep_ in ("appos", "dobj", "attr", "dep") and token.pos_ in ("NOUN", "PROPN"):
                    patient_token = token
                    break

        object_str = None
        if patient_token:
            compounds = [c for c in patient_token.children if c.dep_ in ("compound", "amod") and c.i < patient_token.i]
            if compounds:
                object_str = " ".join([c.text for c in compounds] + [patient_token.text])
            else:
                object_str = patient_token.text

        # Extract Modifiers
        modifiers = []
        for token in doc:
            if token.dep_ == "prep" or token.pos_ == "ADP":
                prep_lemma = token.lemma_.lower()
                pobj = [child for child in token.children if child.dep_ in ("pobj", "dobj")]
                if not pobj:
                    for next_tok in doc[token.i + 1:]:
                        if next_tok.pos_ in ("NOUN", "PROPN"):
                            pobj = [next_tok]
                            break
                if pobj:
                    pobj_tok = pobj[0]
                    pobj_text = pobj_tok.text
                    full_span = doc[token.i : pobj_tok.i + 1].text
                    modifiers.append({
                        "type": "prepositional_phrase",
                        "prep": prep_lemma,
                        "pobj": pobj_text,
                        "text": full_span,
                    })

        verb_str = root_token.text if root_token else None
        return SVOResult(subject=subject_str, verb=verb_str, object=object_str, modifiers=modifiers)

    def parse_sentence(self, text: str, domain_context: Optional[str] = None) -> QuantaGraph:
        """Parses a single natural language sentence into a validated QuantaGraph ASG."""
        doc = self.parse_dependency_tree(text)
        graph = QuantaGraph()

        # Disambiguate verb / root tokens (e.g. 'bit' misclassified as NOUN)
        known_verb_lemmas = {
            "bit": "bite", "bites": "bite", "bite": "bite",
            "chased": "chase", "chases": "chase", "chase": "chase",
            "ran": "run", "runs": "run", "run": "run",
            "saw": "see", "sees": "see", "see": "see",
            "gave": "give", "gives": "give", "give": "give",
            "walked": "walk", "walks": "walk", "walk": "walk",
            "thought": "think", "thinks": "think", "think": "think",
            "knew": "know", "knows": "know", "know": "know",
            "wanted": "want", "wants": "want", "want": "want",
            "felt": "feel", "feels": "feel", "feel": "feel",
            "touched": "touch", "touches": "touch", "touch": "touch",
        }

        # Find main predicate verb (ROOT)
        root_token = None
        for token in doc:
            if token.dep_ == "ROOT":
                root_token = token
                break

        # If ROOT token is misclassified (e.g. noun 'bit'), find best verb token
        if root_token is not None and root_token.pos_ != "VERB":
            for token in doc:
                if token.text.lower() in known_verb_lemmas or token.pos_ == "VERB":
                    root_token = token
                    break

        if root_token is None and len(doc) > 0:
            root_token = doc[0]

        # 1. Construct the Root Predicate Node
        root_node = self._create_root_predicate_node(root_token, doc, domain_context)
        graph.add_node(root_node, set_as_root=True)

        # 2. Extract Agent / Subject (nsubj / nsubjpass / csubj)
        agent_token = None
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass", "csubj") and (token.head == root_token or token.i < root_token.i):
                agent_token = token
                break

        if agent_token is None:
            # Fallback for subject before root verb
            for token in doc:
                if token.i < root_token.i and token.pos_ in ("NOUN", "PROPN", "PRON") and token.dep_ not in ("prep", "pobj", "det"):
                    agent_token = token
                    break

        agent_node = None
        if agent_token:
            # Check for multi-word compounds (e.g. golden retriever)
            compounds = [c for c in agent_token.children if c.dep_ in ("compound", "amod") and c.i < agent_token.i]
            if not compounds:
                # Check preceding sibling tokens before agent
                prev_tokens = [doc[i] for i in range(max(0, agent_token.i - 2), agent_token.i) if doc[i].pos_ in ("ADJ", "NOUN") and doc[i].dep_ not in ("det", "prep")]
                compounds = prev_tokens

            if compounds:
                full_text = " ".join([c.text for c in compounds] + [agent_token.text])
                try:
                    concept = self.grounder.ground_synset(full_text.lower().replace(" ", "_"))
                    agent_node = QuantaNode(vector=concept.vector, anchor=concept.synset_name, literal=full_text)
                except Exception:
                    agent_node = self._create_entity_node(agent_token)
            else:
                agent_node = self._create_entity_node(agent_token)

            # Apply descriptors modifying agent before finalizing CID
            for token in doc:
                if token.pos_ == "ADJ" and (token.head == agent_token or token.i < agent_token.i):
                    self._apply_descriptor_to_node(agent_node, token.lemma_.lower())

            agent_node.set_slot("VAL_X1_AGENT", 1)
            agent_node.set_slot("ROLE_AGENT_CAPABLE", 1)
            root_node.set_slot("VAL_X1_AGENT", 1)
            graph.add_node(agent_node)
            graph.add_edge(root_node, "VAL_X1_AGENT", agent_node)

        # 3. Extract Patient / Object / Attribute (dobj / attr / oprd / acomp)
        patient_token = None
        for token in doc:
            if token.dep_ in ("dobj", "attr", "dative", "acomp", "oprd") and (token.head == root_token or token.i > root_token.i):
                patient_token = token
                break

        if patient_token is None:
            # Fallback for patient if tagged as appos or direct argument after root
            for token in doc:
                if token.i > root_token.i and token.dep_ in ("appos", "dobj", "attr", "dep") and token.pos_ in ("NOUN", "PROPN"):
                    patient_token = token
                    break

        patient_node = None
        if patient_token:
            compounds = [c for c in patient_token.children if c.dep_ in ("compound", "amod") and c.i < patient_token.i]
            if compounds:
                full_text = " ".join([c.text for c in compounds] + [patient_token.text])
                try:
                    concept = self.grounder.ground_synset(full_text.lower().replace(" ", "_"))
                    patient_node = QuantaNode(vector=concept.vector, anchor=concept.synset_name, literal=full_text)
                except Exception:
                    patient_node = self._create_entity_node(patient_token)
            else:
                patient_node = self._create_entity_node(patient_token)

            # Apply descriptors modifying patient before finalizing CID
            for token in doc:
                if token.pos_ == "ADJ" and (token.head == patient_token or (token.i > root_token.i and token.i < patient_token.i)):
                    self._apply_descriptor_to_node(patient_node, token.lemma_.lower())

            patient_node.set_slot("VAL_X2_PATIENT", 1)
            root_node.set_slot("VAL_X2_PATIENT", 1)
            graph.add_node(patient_node)
            graph.add_edge(root_node, "VAL_X2_PATIENT", patient_node)

        # 4. Extract Prepositional Phrases across the entire clause (Destination, Source, Location, Instrument, Manner)
        for token in doc:
            if token.dep_ == "prep" or token.pos_ == "ADP":
                prep_lemma = token.lemma_.lower()
                pobj = [child for child in token.children if child.dep_ in ("pobj", "dobj")]
                if not pobj:
                    # Look ahead for following noun token
                    for next_tok in doc[token.i + 1:]:
                        if next_tok.pos_ in ("NOUN", "PROPN"):
                            pobj = [next_tok]
                            break
                if not pobj:
                    continue

                pobj_token = pobj[0]
                prep_node = self._create_entity_node(pobj_token)

                if prep_lemma in ("to", "into", "towards"):
                    if prep_node.get_slot("TYPE_ANIMATE") == 1 or prep_node.get_slot("TYPE_HUMAN") == 1:
                        prep_node.set_slot("VAL_EXPERIENCER", 1)
                        root_node.set_slot("VAL_EXPERIENCER", 1)
                        graph.add_node(prep_node)
                        graph.add_edge(root_node, "VAL_EXPERIENCER", prep_node)
                    else:
                        prep_node.set_slot("VAL_X3_DESTINATION", 1)
                        prep_node.set_slot("TYPE_SPATIAL_REGION", 1)
                        root_node.set_slot("VAL_X3_DESTINATION", 1)
                        graph.add_node(prep_node)
                        graph.add_edge(root_node, "VAL_X3_DESTINATION", prep_node)
                elif prep_lemma in ("from", "out", "off"):
                    prep_node.set_slot("VAL_X4_SOURCE", 1)
                    prep_node.set_slot("TYPE_SPATIAL_REGION", 1)
                    root_node.set_slot("VAL_X4_SOURCE", 1)
                    graph.add_node(prep_node)
                    graph.add_edge(root_node, "VAL_X4_SOURCE", prep_node)
                elif prep_lemma in ("in", "inside", "at", "on", "within"):
                    prep_node.set_slot("NSM_INSIDE", 1)
                    prep_node.set_slot("TYPE_SPATIAL_REGION", 1)
                    prep_node.set_slot("VAL_LOCATION_SLOT", 1)
                    root_node.set_slot("VAL_LOCATION_SLOT", 1)
                    graph.add_node(prep_node)
                    graph.add_edge(root_node, "VAL_LOCATION_SLOT", prep_node)
                elif prep_lemma in ("with", "by", "using"):
                    prep_node.set_slot("ROLE_INSTRUMENT_USABLE", 1)
                    prep_node.set_slot("VAL_X5_INSTRUMENT", 1)
                    root_node.set_slot("VAL_X5_INSTRUMENT", 1)
                    graph.add_node(prep_node)
                    graph.add_edge(root_node, "VAL_X5_INSTRUMENT", prep_node)
                elif prep_lemma in ("for", "because", "since"):
                    prep_node.set_slot("VAL_PURPOSE_SLOT", 1)
                    root_node.set_slot("VAL_PURPOSE_SLOT", 1)
                    graph.add_node(prep_node)
                    graph.add_edge(root_node, "VAL_PURPOSE_SLOT", prep_node)

        # 5. Extract Adverbial Modifiers & Clauses (Manner, Purpose, Result)
        for token in doc:
            if token.dep_ == "advmod" and token.head == root_token:
                adv_lemma = token.lemma_.lower()
                root_node.set_slot("VAL_MANNER_SLOT", 1)
                if adv_lemma in ("fast", "quickly", "rapidly", "accelerating"):
                    root_node.set_slot("NSM_ACCELERATING_RATE", 1)
            elif token.dep_ == "advcl" and token.head == root_token:
                root_node.set_slot("GRAPH_IS_SUB_EXP", 1)
                root_node.set_slot("VAL_PURPOSE_SLOT", 1)
                root_node.set_slot("VAL_RESULT_SLOT", 1)

        # 6. Extract Predicate Adjectives (e.g. 'A dog was not big')
        for token in doc:
            if token.pos_ == "ADJ":
                is_agent_adj = agent_token and (token.head == agent_token or token.i < agent_token.i)
                is_patient_adj = patient_token and (token.head == patient_token or (token.i > root_token.i and token.i < patient_token.i))
                if not is_agent_adj and not is_patient_adj:
                    adj_lemma = token.lemma_.lower()
                    self._apply_descriptor_to_node(root_node, adj_lemma)

        # 7. Extract Comprehensive Logic, Pronouns, Substantives & Space/Time Primes
        self._apply_comprehensive_linguistic_primes(root_node, doc)
        root_node.compute_cid()

        return graph

    def sentence_to_vector(self, text: str, domain_context: Optional[str] = None) -> QuantaVector:
        """Helper to parse sentence and return the 256-d vector aggregating the entire proposition ASG."""
        graph = self.parse_sentence(text, domain_context=domain_context)
        return graph.to_proposition_vector()

    def _is_motion_verb(self, lemma: str) -> bool:
        if lemma in self.motion_verbs:
            return True
        if NLTK_WN:
            try:
                syns = wn.synsets(lemma, pos=wn.VERB)
                if syns and syns[0].lexname() == "verb.motion":
                    return True
            except Exception:
                pass
        return False

    def _create_root_predicate_node(
        self,
        root_token: Any,
        doc: Any,
        domain_context: Optional[str] = None,
    ) -> QuantaNode:
        """Constructs and populates the root predicate QuantaNode with 4-valued polarity."""
        node = QuantaNode(literal=doc.text)
        lemma = root_token.lemma_.lower()
        if root_token.text.lower() == "bit":
            lemma = "bite"
        elif root_token.text.lower() in ("thinks", "think"):
            lemma = "think"
        elif root_token.text.lower() in ("wants", "want"):
            lemma = "want"
        elif root_token.text.lower() in ("saw", "sees", "see"):
            lemma = "see"

        # Set default literal modality and proposition type
        node.set_slot("MODALITY_LITERAL", 1)
        node.set_slot("GRAPH_ROOT_NODE", 1)

        # Clause polarity
        has_negation = any(t.dep_ == "neg" or t.lemma_.lower() in ("not", "never", "no", "neither", "none", "without", "cannot", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't") for t in doc)
        has_uncertainty = any(w in doc.text.lower() for w in ("maybe", "perhaps", "possibly", "possible", "might", "could", "may", "uncertain", "unclear", "doubt", "suppose", "hypothetical", "whether", "guess", "wonder", "probably", "likely")) or doc.text.strip().endswith("?")

        polarity = 1
        if has_negation:
            polarity = 2
        elif has_uncertainty:
            polarity = 3

        # Band 0 & Band 2: Map Lemma to NSM Primes and Cognitive/Theory-of-Mind Slots
        if self._is_motion_verb(lemma):
            node.set_slot("NSM_MOVE", polarity)
            node.set_slot("TYPE_EVENT", 1)
        elif lemma in self.speech_verbs:
            node.set_slot("NSM_SAY", polarity)
            node.set_slot("NSM_WORDS", polarity)
            node.set_slot("TYPE_COMMUNICATION_MSG", 1)
            node.set_slot("ROLE_COMMUNICATOR", 1)
            node.set_slot("ROLE_EPISTEMIC_AUTHORITY", 1)
            node.set_slot("WN_COMMUNICATION_INFO", 1)
        elif lemma in self.cognition_verbs:
            node.set_slot("NSM_THINK", polarity)
            if lemma == "know":
                node.set_slot("NSM_KNOW", polarity)
            node.set_slot("TYPE_STATE", 1)
            node.set_slot("TOM_BELIEF_FIRST_ORDER", polarity)
            node.set_slot("ROLE_COGNITIVE_SUBJECT", 1)
            node.set_slot("WN_COGNITION_THOUGHT", 1)
        elif lemma in self.volition_verbs:
            node.set_slot("NSM_WANT", polarity)
            node.set_slot("TYPE_STATE", 1)
            node.set_slot("TOM_INTENTION", polarity)
            node.set_slot("TOM_DESIRE", polarity)
            node.set_slot("ROLE_VOLITIONAL_SOURCE", 1)
        elif lemma in self.perception_verbs:
            node.set_slot("NSM_SEE", polarity)
            node.set_slot("TYPE_EVENT", 1)
            node.set_slot("VAL_EXPERIENCER", 1)
            node.set_slot("ROLE_AFFECTIVE_TARGET", 1)
        elif lemma in self.auditory_verbs:
            node.set_slot("NSM_HEAR", polarity)
            node.set_slot("TYPE_EVENT", 1)
            node.set_slot("VAL_EXPERIENCER", 1)
        elif lemma in self.contact_verbs:
            node.set_slot("NSM_DO", polarity)
            node.set_slot("NSM_TOUCH", polarity)
            node.set_slot("TYPE_EVENT", 1)
        elif lemma in self.possession_verbs:
            node.set_slot("NSM_HAVE", polarity)
            node.set_slot("TYPE_STATE", 1)
            node.set_slot("WN_POSSESSION_ASSET", 1)
        elif lemma in self.event_verbs:
            node.set_slot("NSM_HAPPEN", polarity)
            node.set_slot("TYPE_EVENT", 1)
            node.set_slot("WN_EVENT_OCCURRENCE", 1)
        elif lemma in self.life_verbs:
            if lemma in ("live", "breathe", "survive"):
                node.set_slot("NSM_LIVE", polarity)
            elif lemma in ("die", "perish"):
                node.set_slot("NSM_DIE", 1)
            elif lemma == "born":
                node.set_slot("NSM_BORN", 1)
            elif lemma in ("grow", "sprout"):
                node.set_slot("NSM_GROW", polarity)
            node.set_slot("TYPE_EVENT", 1)
        else:
            if root_token.pos_ == "VERB":
                node.set_slot("NSM_DO", polarity)
                node.set_slot("TYPE_EVENT", 1)
            else:
                node.set_slot("TYPE_STATE", 1)

        node.anchor = f"wn:{lemma}.v.01"

        # Band 1: Tense Detection
        morph = str(root_token.morph)
        has_past_aux = any(t.text.lower() in ("did", "didn't", "was", "were", "had") for t in doc)
        if "Tense=Past" in morph or root_token.tag_ in ("VBD", "VBN") or root_token.text.lower() in ("bit", "chased", "saw", "gave", "ran", "thought", "felt", "was", "were", "had", "wanted") or has_past_aux:
            node.set_slot("LJB_PU_PAST_TENSE", 1)
        elif "Tense=Pres" in morph or root_token.tag_ in ("VBP", "VBZ") or root_token.text.lower() in ("thinks", "wants", "bites", "chases", "sees", "runs", "gives", "is", "are", "has"):
            node.set_slot("LJB_CA_PRESENT_TENSE", 1)

        for child in root_token.children:
            if child.dep_ == "aux" and child.lemma_.lower() in ("will", "shall"):
                node.set_slot("LJB_BA_FUTURE_TENSE", 1)

        # Band 3: Epistemic context based on domain and polarity
        if domain_context == "FOLIO":
            node.set_slot("EPIST_DEDUCTIVE_INFERENCE", 1)
            node.set_slot("SOLVER_PROOF_VALIDATED", 1)
        elif domain_context == "ProofWriter":
            node.set_slot("EPIST_AXIOMATIC_PREMISE", 1)
            node.set_slot("SOLVER_CWA_CLOSED_WORLD", 1)
        elif domain_context == "bAbI":
            node.set_slot("EPIST_DIRECT_OBSERVATION", 1)
        else:
            node.set_slot("EPIST_DIRECT_OBSERVATION", 1)

        return node

    def _create_entity_node(self, token: Any) -> QuantaNode:
        """Constructs an entity QuantaNode grounded through WordNet."""
        lemma = token.lemma_.lower()
        try:
            concept = self.grounder.ground_synset(lemma)
            node = QuantaNode(vector=concept.vector, anchor=concept.synset_name, literal=token.text)
        except Exception:
            # Fallback when not found in WordNet
            node = QuantaNode(literal=token.text)
            if token.ent_type_ == "PERSON" or token.text.istitle():
                node.set_slot("TYPE_HUMAN", 1)
                node.set_slot("TYPE_ANIMATE", 1)
                node.set_slot("ROLE_AGENT_CAPABLE", 1)
                node.set_slot("ROLE_SENTIENT", 1)
                node.set_slot("WN_PERSON_HUMAN", 1)
                node.anchor = f"wn:person.n.01"
            else:
                node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
                node.anchor = f"wn:{lemma}.n.01"

        return node

    def _apply_descriptor_to_node(self, node: QuantaNode, adj_lemma: str, polarity: int = 1):
        """Applies qualitative descriptors and evaluators to QuantaNode with quaternary polarity."""
        node.set_slot("WN_ATTRIBUTE_PROP", 1)
        node.set_slot("TYPE_ATTRIBUTE_PROPERTY", 1)

        if adj_lemma in ("good", "great", "fine", "excellent", "nice", "positive", "beneficial"):
            node.set_slot("NSM_GOOD", polarity)
        elif adj_lemma in ("bad", "terrible", "poor", "wrong", "negative", "evil", "harmful"):
            node.set_slot("NSM_BAD", polarity)
        elif adj_lemma in ("big", "large", "huge", "giant", "vast", "heavy", "immense"):
            node.set_slot("NSM_BIG", polarity)
        elif adj_lemma in ("small", "little", "tiny", "minor", "compact", "mini"):
            node.set_slot("NSM_SMALL", polarity)
        elif adj_lemma in ("quiet", "silent", "calm"):
            node.set_slot("NSM_HEAR", 2 if polarity == 1 else 1)
        elif adj_lemma in ("cold", "freezing"):
            node.set_slot("NSM_FEEL", polarity)
        elif adj_lemma in ("rough", "hard", "tough"):
            node.set_slot("NSM_TOUCH", polarity)

    def _apply_comprehensive_linguistic_primes(self, node: QuantaNode, doc: Any):
        """Extracts universal NSM primes, quantifiers, negation, spatial/temporal relations, and pronouns with 4-valued polarity."""
        text_lower = doc.text.lower()
        tokens = [t.text.lower() for t in doc]
        lemmas = [t.lemma_.lower() for t in doc]

        # Determine clause polarity: 1=True, 2=Negated, 3=Uncertain
        has_negation = any(t.dep_ == "neg" or t.lemma_.lower() in ("not", "never", "no", "neither", "none", "without", "cannot", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't") for t in doc)
        has_uncertainty = any(w in text_lower for w in ("maybe", "perhaps", "possibly", "possible", "might", "could", "may", "uncertain", "unclear", "doubt", "suppose", "hypothetical", "whether", "guess", "wonder", "probably", "likely")) or doc.text.strip().endswith("?")

        # 1. Pronouns & Substantives
        if any(w in tokens for w in ("i", "me", "my", "myself")):
            node.set_slot("NSM_I", 1)
        if any(w in tokens for w in ("you", "your", "yours", "yourself", "thee", "thou")):
            node.set_slot("NSM_YOU", 1)
        if any(w in tokens for w in ("someone", "somebody", "anyone", "anybody")):
            node.set_slot("NSM_SOMEONE", 1)
        if any(w in tokens for w in ("something", "anything", "object", "item", "entity")):
            node.set_slot("NSM_SOMETHING", 1)
        if any(w in tokens for w in ("people", "humans", "folks", "crowd", "citizens", "society")):
            node.set_slot("NSM_PEOPLE", 1)
        if any(w in tokens for w in ("body", "flesh", "torso", "skin", "organism", "corpse", "arm", "leg", "hand", "head", "eye")):
            node.set_slot("NSM_BODY", 1)
            node.set_slot("WN_BODY_PART", 1)
        if any(w in tokens for w in ("this", "these", "that", "those", "the")):
            node.set_slot("NSM_THIS", 1)
        if any(w in tokens for w in ("same", "identical", "equal", "equivalent", "like", "as", "resembles", "similar")):
            node.set_slot("NSM_SAME", 2 if has_negation else 1)
        if any(w in tokens for w in ("other", "another", "different", "else", "distinct", "alternate", "various")):
            node.set_slot("NSM_OTHER", 1)

        # 2. Quantifiers, Counting & Order
        if any(w in tokens for w in ("one", "1", "single", "lone", "individual", "a", "an")):
            node.set_slot("NSM_ONE", 1)
        if any(w in tokens for w in ("two", "2", "pair", "both", "couple", "dual")):
            node.set_slot("NSM_TWO", 1)
            node.set_slot("LJB_MEI_CARDINAL", 1)
        if any(w in tokens for w in ("some", "exist", "exists", "several")):
            node.set_slot("NSM_SOME", 1)
            node.set_slot("LJB_SUO_AT_LEAST_ONE", 1)
        if any(w in tokens for w in ("all", "every", "each", "universal", "entire", "any")):
            node.set_slot("NSM_ALL", 1)
            node.set_slot("LJB_RO_ALL_QUANT", 1)
            node.set_slot("EPIST_INDUCTIVE_GENERAL", 1)
        if any(w in tokens for w in ("much", "many", "lot", "lots", "numerous", "multitude", "plenty")):
            node.set_slot("NSM_MUCH", 1)
        if any(w in tokens for w in ("little", "slight", "scant")):
            node.set_slot("NSM_LITTLE", 1)
        if any(w in tokens for w in ("few", "fewer", "rare", "seldom")):
            node.set_slot("NSM_FEW", 1)
        if any(w in tokens for w in ("more", "greater", "longer", "higher", "larger", "older", "better", "faster")):
            node.set_slot("NSM_MORE", 1)
        if any(w in tokens for w in ("part", "portion", "piece", "component", "fraction", "segment", "element", "member")):
            node.set_slot("NSM_PART", 1)
            node.set_slot("MEREOLOGY_MERONYM_PART", 1)
        if any(w in tokens for w in ("first", "second", "third", "next", "last", "1st", "2nd", "3rd", "finally")):
            node.set_slot("LJB_MOI_ORDINAL", 1)
            node.set_slot("TEMP_ALLEN_BEFORE", 1)
        if any(w in tokens for w in ("each other", "one another", "mutually", "reciprocally", "together")):
            node.set_slot("LJB_SOI_RECIPROCAL", 1)

        # 3. Augmentatives & Feelings
        if any(w in tokens for w in ("very", "extremely", "highly", "really", "quite", "super", "deeply")):
            node.set_slot("NSM_VERY", 1)
        if any(w in tokens for w in ("feel", "feels", "felt", "feeling", "pain", "pleasure", "sad", "happy", "glad", "fear", "afraid", "angry", "love", "hate")):
            node.set_slot("NSM_FEEL", 2 if has_negation else (3 if has_uncertainty else 1))
            node.set_slot("WN_FEELING_EMOTION", 1)
            node.set_slot("ROLE_AFFECTIVE_TARGET", 1)
        if any(w in tokens for w in ("happen", "happens", "happened", "occur", "occurs", "occurred", "transpire", "transpired")):
            node.set_slot("NSM_HAPPEN", 2 if has_negation else 1)
            node.set_slot("WN_EVENT_OCCURRENCE", 1)
        if any(w in tokens for w in ("located", "situated", "stationed", "placed", "positioned", "resides", "dwells")):
            node.set_slot("NSM_BE_SOMEWHERE", 2 if has_negation else (3 if has_uncertainty else 1))
        if "there is" in text_lower or "there are" in text_lower or "there was" in text_lower or "there were" in text_lower or "exists" in tokens or "exist" in tokens:
            node.set_slot("NSM_THERE_IS", 2 if has_negation else 1)

        # Vitality
        if any(w in tokens for w in ("live", "lives", "lived", "living", "alive", "breathe", "survive", "survives")):
            node.set_slot("NSM_LIVE", 2 if has_negation else 1)
        if any(w in tokens for w in ("die", "dies", "died", "dead", "death", "perish", "kill", "killed", "fatal", "lethal")):
            node.set_slot("NSM_DIE", 1)
        if any(w in tokens for w in ("born", "birth", "conceived", "originated", "mother", "father", "son", "daughter", "child", "parent", "sister", "brother")):
            node.set_slot("NSM_BORN", 1)
        if any(w in tokens for w in ("grow", "grows", "grew", "growing", "expand", "develop", "increase", "mature", "sprout")):
            node.set_slot("NSM_GROW", 2 if has_negation else 1)

        # 4. Temporal Primes & Durations
        if any(w in tokens for w in ("now", "currently", "today", "present", "nowadays", "at present")):
            node.set_slot("NSM_NOW", 1)
        if any(w in tokens for w in ("before", "prior", "earlier", "previously", "yesterday", "past", "formerly")):
            node.set_slot("NSM_BEFORE", 1)
            if any(w in tokens for w in ("recently", "just now", "shortly ago")):
                node.set_slot("LJB_ZI_SHORT_PAST", 1)
            elif any(w in tokens for w in ("ancient", "long ago", "centuries ago")):
                node.set_slot("LJB_ZU_LONG_PAST", 1)
            else:
                node.set_slot("LJB_ZA_MEDIUM_PAST", 1)
        if any(w in tokens for w in ("after", "then", "later", "subsequently", "next", "tomorrow", "future")):
            node.set_slot("NSM_AFTER", 1)
        if any(w in tokens for w in ("always", "forever", "eternal", "decades", "centuries", "years", "long")):
            node.set_slot("NSM_A_LONG_TIME", 1)
        if any(w in tokens for w in ("brief", "briefly", "shortly", "quick", "quickly", "fast", "soon", "swiftly")):
            node.set_slot("NSM_A_SHORT_TIME", 1)
        if any(w in tokens for w in ("while", "during", "period", "duration", "hours", "days", "months")):
            node.set_slot("NSM_FOR_SOME_TIME", 1)
        if any(w in tokens for w in ("moment", "instant", "second", "flash", "immediate", "immediately")):
            node.set_slot("NSM_MOMENT", 1)
        if any(w in tokens for w in ("when", "time", "hour", "day", "year", "date")):
            node.set_slot("VAL_TIME_SLOT", 1)
        if any(w in tokens for w in ("interval", "span", "period", "duration", "era", "century", "decade", "season")):
            node.set_slot("TYPE_TEMPORAL_INTERVAL", 1)

        # 5. Spatial Primes & Relations
        if any(w in tokens for w in ("here", "present", "nearby", "hither")):
            node.set_slot("NSM_HERE", 2 if has_negation else 1)
            node.set_slot("LJB_VI_SHORT_DISTANCE", 1)
        if any(w in tokens for w in ("above", "over", "top", "up", "on", "upon", "overhead")):
            node.set_slot("NSM_ABOVE", 1)
        if any(w in tokens for w in ("below", "under", "beneath", "down", "bottom", "underneath")):
            node.set_slot("NSM_BELOW", 1)
        if any(w in tokens for w in ("far", "distant", "away", "remote", "beyond")):
            node.set_slot("NSM_FAR", 1)
            node.set_slot("LJB_VU_LONG_DISTANCE", 1)
        if any(w in tokens for w in ("near", "close", "adjacent", "by", "beside", "next to", "alongside")):
            node.set_slot("NSM_NEAR", 1)
            node.set_slot("LJB_VI_SHORT_DISTANCE", 1)
            node.set_slot("SPATIAL_RCC_EXT_CONNECTED", 1)
        if any(w in tokens for w in ("side", "left", "right", "flank", "edge", "border")):
            node.set_slot("NSM_SIDE", 1)
        if any(w in tokens for w in ("inside", "in", "within", "into", "interior")):
            node.set_slot("NSM_INSIDE", 2 if has_negation else 1)
        if any(w in tokens for w in ("outside", "out", "exterior", "outdoors")):
            node.set_slot("NSM_OUTSIDE", 1)
        if any(w in tokens for w in ("between", "among", "amid", "middle", "center")):
            node.set_slot("NSM_BETWEEN", 1)
        if any(w in tokens for w in ("touch", "touching", "contact", "connected", "attached", "against")):
            node.set_slot("NSM_TOUCHING", 2 if has_negation else 1)
        if any(w in tokens for w in ("where", "place", "location", "area", "zone", "region", "garden", "room", "hallway", "kitchen", "bathroom", "office", "bedroom")):
            node.set_slot("VAL_LOCATION_SLOT", 3 if has_uncertainty else 1)

        # 6. Concurrency & Synchronization
        if any(w in tokens for w in ("concurrent", "parallel", "simultaneously", "meanwhile", "async")):
            node.set_slot("LJB_ASYNC_CONCURRENT", 1)
        if any(w in tokens for w in ("lock", "locked", "mutex", "blocked", "await", "wait", "depends")):
            node.set_slot("LJB_MUTEX_DEPENDENCY", 1)
        if any(w in tokens for w in ("race", "conflict", "contention", "clash", "interfere")):
            node.set_slot("LJB_RACE_CONDITION", 1)

        # 7. Logical Connectives, Negation & Modality
        if has_negation:
            node.set_slot("LJB_NA_NEGATION", 2)  # Quaternary 2: Explicitly Negated
        else:
            node.set_slot("NSM_TRUE", 1)

        if has_uncertainty:
            node.set_slot("NSM_MAYBE", 3)          # Quaternary 3: Uncertain / Modal
            node.set_slot("MODALITY_HYPOTHETICAL", 3)
            node.set_slot("EPIST_PROB_MARGINAL", 3)
            node.set_slot("EPIST_FUZZY_PLAUSIBILITY", 3)
            if doc.text.strip().endswith("?"):
                node.set_slot("GRAPH_QUERY_TARGET", 3)
                node.set_slot("TYPE_PROPOSITION", 3)
        else:
            node.set_slot("EPIST_PROB_CERTAIN", 1)

        if any(w in tokens for w in ("none", "neither", "no", "nobody", "nothing")):
            node.set_slot("LJB_NO_NONE_QUANT", 1)

        if any(w in tokens for w in ("and", "plus", "as well as", "along with", "also")):
            node.set_slot("LJB_JE_AND", 1)
        if any(w in tokens for w in ("or", "alternatively")):
            node.set_slot("LJB_JA_OR", 1)
        if "either" in tokens or "xor" in tokens:
            node.set_slot("LJB_JON_XOR", 1)
        if any(w in tokens for w in ("if", "suppose", "assuming", "whether", "provided")):
            node.set_slot("LJB_GANAI_IF_THEN", 3 if has_uncertainty else 1)
        if any(w in tokens for w in ("because", "since", "due to", "explains", "causes")):
            node.set_slot("LJB_GANAI_IF_THEN", 1)
            node.set_slot("EPIST_ABDUCTIVE_BEST_EXPL", 1)
            node.set_slot("WN_MOTIVE_REASON", 1)
            node.set_slot("CAUSAL_DIRECT_MECHANISM", 1)
        if any(w in tokens for w in ("like", "as", "similar", "resembles")):
            node.set_slot("NSM_SAME", 1)

        # 8. Modal Auxiliaries & Deontics
        if any(w in tokens for w in ("must", "obliged", "required", "shall", "ought", "need", "needs", "need to")):
            if has_negation:
                node.set_slot("EPIST_DEONTIC_PROHIBITION", 1)
            else:
                node.set_slot("EPIST_DEONTIC_OBLIGATION", 1)
        if any(w in tokens for w in ("may", "permitted", "allowed", "can", "able", "capable")):
            node.set_slot("EPIST_DEONTIC_PERMISSION", 1)
            node.set_slot("NSM_CAN", 1)
        if any(w in tokens for w in ("maybe", "might", "possibly", "perhaps", "could")):
            node.set_slot("NSM_MAYBE", 3)
            node.set_slot("EPIST_PROB_MARGINAL", 3)
        if any(w in tokens for w in ("likely", "probably", "expected", "usually", "tend")):
            node.set_slot("EPIST_PROB_HIGH", 1)
            node.set_slot("EPIST_STATISTICAL_EDGE", 1)
        if any(w in tokens for w in ("said", "told", "heard", "reported", "claimed", "testified")):
            node.set_slot("EPIST_HEARSAY_TESTIMONY", 1)

        # 9. Natural, Process & Collection Ontologies
        if any(w in tokens for w in ("rock", "stone", "water", "tree", "plant", "flower", "sun", "moon", "star", "mountain", "river", "soil", "forest", "sea")):
            node.set_slot("TYPE_NATURAL_OBJECT", 1)
            node.set_slot("WN_OBJECT_NATURAL", 1)
        if any(w in tokens for w in ("collection", "set", "group", "family", "class", "bunch", "series", "array", "list")):
            node.set_slot("TYPE_COLLECTION_SET", 1)
            node.set_slot("WN_GROUP_SOCIAL", 1)
        if any(w in tokens for w in ("procedure", "process", "series", "cycle", "protocol", "step", "method", "routine", "program")):
            node.set_slot("WN_PROCESS_SERIES", 1)
            node.set_slot("TYPE_PROCESS", 1)

        # 10. Discourse, Anaphora, Identity, Rate & State Transitions
        if any(t.pos_ == "PRON" for t in doc):
            node.set_slot("GRAPH_ANAPHORA_TARGET", 1)
            node.set_slot("GRAPH_COREF_BUNDLE", 1)
        if any(t.pos_ == "NUM" or t.like_num for t in doc):
            node.set_slot("TYPE_NUMERIC_VALUE", 1)
            node.set_slot("TYPE_MEASURE_SCALAR", 1)
        if any(w in tokens for w in ("is", "are", "was", "were", "be", "equal", "equals", "means", "identical", "same as")):
            node.set_slot("LJB_DU_IDENTITY", 1)
        if any(w in tokens for w in ("all", "every", "each", "always", "normally", "usually", "generally", "typically")):
            node.set_slot("EPIST_DEFAULT_HEURISTIC", 1)
        if any(w in tokens for w in ("always", "constantly", "continually", "continuously", "regularly", "steadily", "keeps", "kept")):
            node.set_slot("NSM_CONTINUOUS_RATE", 1)
        if any(w in tokens for w in ("then", "after", "before", "next", "subsequently", "followed", "following")):
            node.set_slot("GRAPH_ORDERED_SEQ", 1)
        if any(w in tokens for w in ("if", "implies", "whenever")):
            node.set_slot("GRAPH_BRANCH_COND", 1)
            node.set_slot("GRAPH_BRANCH_THEN", 1)
        if any(w in tokens for w in ("moved", "went", "travelled", "journeyed", "walked", "entered", "left", "dropped", "picked", "got", "took")):
            node.set_slot("TEMP_ALLEN_MEETS", 1)
        if any(w in tokens for w in ("want", "wants", "wanted", "wish", "wishes", "intend", "intends", "purpose", "goal")):
            node.set_slot("TOM_INTENTION", 1)
        if any(w in tokens for w in ("meet", "met", "visit", "visited", "talk", "talked", "together", "each other")):
            node.set_slot("TOM_SHARED_ATTENTION", 1)
