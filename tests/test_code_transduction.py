"""Neural Code Discourse Transduction Test Suite (Section 9 / exp-009a).

Verifies the unified neuro-symbolic Mentalese code understanding pipeline:
1. System Prompt Specification: Accepts programming language source code as discourse.
2. Code-Domain Slot Vocabulary: StructuralValue code constructs and Band 7 edge types.
3. Forward Transduction (Python): Functions, calls, returns, control-flow & data-flow edges.
4. Forward Transduction (Java): Classes, inheritance, interfaces, method invocations.
5. Cross-Language Semantic Equivalence: Python vs Java factorial share identical semantic core under lattice meet.
6. Cognitive Pipeline Ingestion: Code ingested into PageTable and ActiveCanvas under O(1) bound.
7. Spreading Activation Code Retrieval: High-speed (< 5ms) caller subgraph extraction without distractors.
8. Neural Reverse Realization: ASG -> valid target-language code.
9. Architecture Principle: Zero language-specific parser imports in the live transduction pipeline.
"""

from __future__ import annotations

import inspect
import time
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_name
from core.types import (
    CALLS,
    CFG_NEXT,
    DATA_FLOW_DEF_USE,
    GRAPH_BRANCH_COND,
    GRAPH_CALL_SITE,
    GRAPH_CLASS_DEF,
    GRAPH_CONTROL_LOOP,
    GRAPH_EXCEPTION_HANDLE,
    GRAPH_FUNCTION_DEF,
    GRAPH_INTERFACE_DEF,
    GRAPH_SCOPED_CONTEXT,
    GRAPH_VARIABLE_BIND,
    IMPLEMENTS,
    IMPORTS,
    INHERITS_FROM,
    StructuralValue,
)
from parser.asg_compiler import ASGCompiler
from parser.sexpr_parser import parse_sexpr, to_sexpr
from parser.unsloth_transducer import (
    DEFAULT_UNSLOTH_SYSTEM_PROMPT,
    MockUnslothTransducer,
    UnslothTransducer,
)
from pipeline.cognitive_pipeline import CognitivePipeline
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from verification.lattice_gate import LatticeInvarianceGate


# ---------------------------------------------------------------------------
# Canonical Deterministic Code Fixtures (Task 9.5)
# ---------------------------------------------------------------------------

PYTHON_TAX_CODE = """def calculate_tax(subtotal: float) -> float:
    rate = get_tax_rate()
    return subtotal * rate"""

PYTHON_TAX_SEXPR = """(graph :chunk-id "chunk_py_tax"
  (entity :id E1 :type ARTIFACT :label "calculate_tax" :surface "def calculate_tax(subtotal: float) -> float:")
  (entity :id E2 :type ARTIFACT :label "subtotal" :surface "subtotal")
  (entity :id E3 :type ARTIFACT :label "rate" :surface "rate")
  (entity :id E4 :type ARTIFACT :label "get_tax_rate" :surface "get_tax_rate()")
  (event :id Ev1 :pred call :agent E1 :patient E4 :time "during execution" :tense PRESENT :polarity TRUE :raw-text "rate = get_tax_rate()")
  (event :id Ev2 :pred return :agent E1 :patient E3 :time "after call" :tense PRESENT :polarity TRUE :raw-text "return subtotal * rate")
  (relation :type CALLS :source E1 :target E4)
  (relation :type CFG_NEXT :source Ev1 :target Ev2)
  (relation :type DATA_FLOW_DEF_USE :source E3 :target Ev2)
)"""

JAVA_PROCESSOR_CODE = """public class OrderProcessor extends BaseProcessor implements IProcessor {
    public void processOrder(Order order) {
        validate(order);
    }
}"""

JAVA_PROCESSOR_SEXPR = """(graph :chunk-id "chunk_java_processor"
  (entity :id E1 :type ARTIFACT :label "OrderProcessor" :surface "public class OrderProcessor")
  (entity :id E2 :type ARTIFACT :label "BaseProcessor" :surface "BaseProcessor")
  (entity :id E3 :type ARTIFACT :label "IProcessor" :surface "IProcessor")
  (entity :id E4 :type ARTIFACT :label "processOrder" :surface "public void processOrder(Order order)")
  (entity :id E5 :type ARTIFACT :label "order" :surface "Order order")
  (entity :id E6 :type ARTIFACT :label "validate" :surface "validate(order)")
  (event :id Ev1 :pred invoke :agent E4 :patient E6 :time "during execution" :tense PRESENT :polarity TRUE :raw-text "validate(order);")
  (relation :type INHERITS_FROM :source E1 :target E2)
  (relation :type IMPLEMENTS :source E1 :target E3)
  (relation :type CALLS :source E4 :target E6)
)"""

PYTHON_FACTORIAL_CODE = """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

PYTHON_FACTORIAL_SEXPR = """(graph :chunk-id "chunk_py_factorial"
  (entity :id E1 :type ARTIFACT :label "factorial" :surface "def factorial(n: int) -> int:")
  (entity :id E2 :type ARTIFACT :label "n" :surface "n")
  (event :id Ev1 :pred branch :agent E1 :patient E2 :time "condition check" :tense PRESENT :polarity TRUE :raw-text "if n <= 1: return 1")
  (event :id Ev2 :pred multiply :agent E1 :patient E2 :time "recursive step" :tense PRESENT :polarity TRUE :raw-text "return n * factorial(n - 1)")
  (relation :type CALLS :source E1 :target E1)
  (relation :type CFG_NEXT :source Ev1 :target Ev2)
)"""

JAVA_FACTORIAL_CODE = """public class MathUtils {
    public static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
}"""

JAVA_FACTORIAL_SEXPR = """(graph :chunk-id "chunk_java_factorial"
  (entity :id E1 :type ARTIFACT :label "factorial" :surface "public static int factorial(int n)")
  (entity :id E2 :type ARTIFACT :label "n" :surface "int n")
  (entity :id E3 :type ARTIFACT :label "MathUtils" :surface "public class MathUtils")
  (event :id Ev1 :pred branch :agent E1 :patient E2 :time "condition check" :tense PRESENT :polarity TRUE :raw-text "if (n <= 1) return 1;")
  (event :id Ev2 :pred multiply :agent E1 :patient E2 :time "recursive step" :tense PRESENT :polarity TRUE :raw-text "return n * factorial(n - 1);")
  (relation :type CALLS :source E1 :target E1)
  (relation :type CFG_NEXT :source Ev1 :target Ev2)
)"""

CALLER_CORPUS_CODE = """class OrderService {
    void handleCheckout(Order order) {
        OrderProcessor processor = new OrderProcessor();
        processor.processOrder(order);
    }
}

class PaymentAuditService {
    void auditTransaction(Order order) {
        OrderProcessor processor = new OrderProcessor();
        processor.processOrder(order);
    }
}

class InventoryService {
    void restockItem(String sku) {
        updateInventory(sku);
    }
}"""

CALLER_CORPUS_SEXPR = """(graph :chunk-id "chunk_code_corpus"
  (entity :id E1 :type ARTIFACT :label "OrderService" :surface "class OrderService")
  (entity :id E2 :type ARTIFACT :label "handleCheckout" :surface "void handleCheckout(Order order)")
  (entity :id E3 :type ARTIFACT :label "PaymentAuditService" :surface "class PaymentAuditService")
  (entity :id E4 :type ARTIFACT :label "auditTransaction" :surface "void auditTransaction(Order order)")
  (entity :id E5 :type ARTIFACT :label "OrderProcessor" :surface "OrderProcessor")
  (entity :id E6 :type ARTIFACT :label "processOrder" :surface "processor.processOrder(order)")
  (entity :id E7 :type ARTIFACT :label "InventoryService" :surface "class InventoryService")
  (entity :id E8 :type ARTIFACT :label "restockItem" :surface "void restockItem(String sku)")
  (entity :id E9 :type ARTIFACT :label "updateInventory" :surface "updateInventory(sku)")
  (event :id Ev1 :pred call :agent E2 :patient E6 :tense PRESENT :polarity TRUE :raw-text "processor.processOrder(order);")
  (event :id Ev2 :pred call :agent E4 :patient E6 :tense PRESENT :polarity TRUE :raw-text "processor.processOrder(order);")
  (event :id Ev3 :pred call :agent E8 :patient E9 :tense PRESENT :polarity TRUE :raw-text "updateInventory(sku);")
  (relation :type CALLS :source E2 :target E6)
  (relation :type CALLS :source E4 :target E6)
  (relation :type CALLS :source E8 :target E9)
)"""


@pytest.fixture
def mock_code_transducer():
    """Fixture providing MockUnslothTransducer pre-seeded with code fixtures."""
    mock = MockUnslothTransducer()
    # Register forward fixtures
    mock.register_fixture(PYTHON_TAX_CODE, PYTHON_TAX_SEXPR)
    mock.register_fixture(JAVA_PROCESSOR_CODE, JAVA_PROCESSOR_SEXPR)
    mock.register_fixture(PYTHON_FACTORIAL_CODE, PYTHON_FACTORIAL_SEXPR)
    mock.register_fixture(JAVA_FACTORIAL_CODE, JAVA_FACTORIAL_SEXPR)
    mock.register_fixture(CALLER_CORPUS_CODE, CALLER_CORPUS_SEXPR)

    # Register realization fixtures
    mock.register_realization_fixture(PYTHON_TAX_SEXPR, PYTHON_TAX_CODE)
    mock.register_realization_fixture(JAVA_PROCESSOR_SEXPR, JAVA_PROCESSOR_CODE)
    mock.register_realization_fixture(PYTHON_FACTORIAL_SEXPR, PYTHON_FACTORIAL_CODE)
    mock.register_realization_fixture(JAVA_FACTORIAL_SEXPR, JAVA_FACTORIAL_CODE)
    return mock


@pytest.fixture
def compiler():
    return ASGCompiler()


# ---------------------------------------------------------------------------
# Task 9.1: System Prompt Specification Tests
# ---------------------------------------------------------------------------

def test_code_domain_system_prompt_spec():
    """Verify Task 9.1: System prompt accepts source code and specifies computational primitives."""
    assert "programming language source code" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "CODE & COMPUTATIONAL DISCOURSE" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "CALLS" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "INHERITS_FROM" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "IMPLEMENTS" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "CFG_NEXT" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "DATA_FLOW_DEF_USE" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "DEMONSTRATION 6 (PYTHON FUNCTION & CALL GRAPH)" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "DEMONSTRATION 7 (JAVA CLASS WITH INHERITANCE & INTERFACE)" in DEFAULT_UNSLOTH_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Task 9.2: Slot Vocabulary Registration Tests
# ---------------------------------------------------------------------------

def test_code_domain_slot_vocabulary():
    """Verify Task 9.2: StructuralValue entries and edge type constants are registered."""
    # Structural slot constants
    assert hasattr(StructuralValue, "GRAPH_INTERFACE_DEF")
    assert hasattr(StructuralValue, "GRAPH_CALL_SITE")
    assert hasattr(StructuralValue, "GRAPH_VARIABLE_BIND")
    assert hasattr(StructuralValue, "GRAPH_CONTROL_LOOP")
    assert hasattr(StructuralValue, "GRAPH_BRANCH_COND")
    assert hasattr(StructuralValue, "GRAPH_EXCEPTION_HANDLE")
    assert hasattr(StructuralValue, "GRAPH_FUNCTION_DEF")
    assert hasattr(StructuralValue, "GRAPH_CLASS_DEF")
    assert hasattr(StructuralValue, "GRAPH_SCOPED_CONTEXT")

    # Edge type constants
    assert CALLS == "CALLS"
    assert INHERITS_FROM == "INHERITS_FROM"
    assert IMPLEMENTS == "IMPLEMENTS"
    assert IMPORTS == "IMPORTS"
    assert CFG_NEXT == "CFG_NEXT"
    assert DATA_FLOW_DEF_USE == "DATA_FLOW_DEF_USE"

    # Canonical slot lookup via slots module
    s_iface = get_slot_by_name("GRAPH_INTERFACE_DEF")
    assert s_iface is not None
    assert s_iface.index == 208  # Aliased to GRAPH_INTERFACE_TRAIT_DEF

    s_call = get_slot_by_name("GRAPH_CALL_SITE")
    assert s_call is not None
    assert s_call.index == 179  # Aliased to GRAPH_INVOCATION_CALL

    s_cls = get_slot_by_name("GRAPH_CLASS_DEF")
    assert s_cls is not None
    assert s_cls.index == 207  # Aliased to GRAPH_CLASS_STRUCT_DEF

    s_loop = get_slot_by_name("GRAPH_CONTROL_LOOP")
    assert s_loop is not None
    assert s_loop.index == 177

    s_branch = get_slot_by_name("GRAPH_BRANCH_COND")
    assert s_branch is not None
    assert s_branch.index == 174


# ---------------------------------------------------------------------------
# Task 9.6.1: Python Function Forward Transduction
# ---------------------------------------------------------------------------

def test_python_forward_transduction(mock_code_transducer, compiler):
    """Verify Python function parses into canonical ASG with calls and data-flow edges."""
    extraction = mock_code_transducer.transduce(text=PYTHON_TAX_CODE)
    assert extraction.chunk_id == "chunk_py_tax"
    assert len(extraction.entities) == 4
    assert len(extraction.events) == 2
    assert len(extraction.relations) == 3

    # Compile to QuantaGraph
    graph = compiler.compile(extraction, validate=False)
    assert len(graph.nodes) >= 6

    # Verify entities
    labels = {n.literal for n in graph.nodes.values() if n.literal}
    assert "calculate_tax" in labels
    assert "get_tax_rate" in labels
    assert "subtotal" in labels
    assert "rate" in labels

    # Verify directed CALLS relation edge
    calc_nodes = [n for n in graph.nodes.values() if n.literal == "calculate_tax"]
    assert len(calc_nodes) == 1
    calc_node = calc_nodes[0]
    tax_rate_nodes = [n for n in graph.nodes.values() if n.literal == "get_tax_rate"]
    assert len(tax_rate_nodes) == 1
    assert "CALLS" in calc_node.edges
    assert tax_rate_nodes[0].cid in calc_node.edges["CALLS"]

    # Verify event sequencing (CFG_NEXT)
    ev1_nodes = [n for n in graph.nodes.values() if n.anchor and "call" in n.anchor]
    assert len(ev1_nodes) == 1
    ev2_nodes = [n for n in graph.nodes.values() if n.anchor and "return" in n.anchor]
    assert len(ev2_nodes) == 1
    assert "CFG_NEXT" in ev1_nodes[0].edges
    assert ev2_nodes[0].cid in ev1_nodes[0].edges["CFG_NEXT"]


# ---------------------------------------------------------------------------
# Task 9.6.2: Java Class with Inheritance & Interface
# ---------------------------------------------------------------------------

def test_java_forward_transduction(mock_code_transducer, compiler):
    """Verify Java class with inheritance & interface maps to canonical ASG."""
    extraction = mock_code_transducer.transduce(text=JAVA_PROCESSOR_CODE)
    assert extraction.chunk_id == "chunk_java_processor"
    assert len(extraction.entities) == 6
    assert len(extraction.events) == 1
    assert len(extraction.relations) == 3

    graph = compiler.compile(extraction, validate=False)

    proc_nodes = [n for n in graph.nodes.values() if n.literal == "OrderProcessor"]
    assert len(proc_nodes) == 1
    proc_node = proc_nodes[0]

    base_nodes = [n for n in graph.nodes.values() if n.literal == "BaseProcessor"]
    assert len(base_nodes) == 1
    iface_nodes = [n for n in graph.nodes.values() if n.literal == "IProcessor"]
    assert len(iface_nodes) == 1

    # Verify INHERITS_FROM and IMPLEMENTS edges
    assert "INHERITS_FROM" in proc_node.edges
    assert base_nodes[0].cid in proc_node.edges["INHERITS_FROM"]

    assert "IMPLEMENTS" in proc_node.edges
    assert iface_nodes[0].cid in proc_node.edges["IMPLEMENTS"]

    # Verify method call site edge
    m_nodes = [n for n in graph.nodes.values() if n.literal == "processOrder"]
    v_nodes = [n for n in graph.nodes.values() if n.literal == "validate"]
    assert len(m_nodes) == 1 and len(v_nodes) == 1
    assert "CALLS" in m_nodes[0].edges
    assert v_nodes[0].cid in m_nodes[0].edges["CALLS"]


# ---------------------------------------------------------------------------
# Task 9.6.3: Cross-Language Structural Equivalence (Python vs Java Factorial)
# ---------------------------------------------------------------------------

def test_cross_language_structural_equivalence(mock_code_transducer, compiler):
    """Verify Python and Java factorial implementations share the same semantic core."""
    ext_py = mock_code_transducer.transduce(text=PYTHON_FACTORIAL_CODE)
    ext_java = mock_code_transducer.transduce(text=JAVA_FACTORIAL_CODE)

    graph_py = compiler.compile(ext_py, validate=False)
    graph_java = compiler.compile(ext_java, validate=False)

    # Both must identify the recursive function node
    py_func = [n for n in graph_py.nodes.values() if n.literal == "factorial"][0]
    java_func = [n for n in graph_java.nodes.values() if n.literal == "factorial"][0]

    # Both must have self-recursive CALLS edge
    assert "CALLS" in py_func.edges
    assert py_func.cid in py_func.edges["CALLS"]
    assert "CALLS" in java_func.edges
    assert java_func.cid in java_func.edges["CALLS"]

    # Verify semantic consistency under LatticeInvarianceGate
    gate = LatticeInvarianceGate()
    is_sound, diff_count, errors = gate.verify_round_trip_invariance(graph_py, graph_java)
    assert is_sound, f"Cross-lingual lattice meet produced contradictions: {errors}"
    assert len(errors) == 0


# ---------------------------------------------------------------------------
# Task 9.6.4: Code-as-Discourse Pipeline Ingestion
# ---------------------------------------------------------------------------

def test_code_as_discourse_pipeline_ingest(mock_code_transducer):
    """Verify CognitivePipeline.ingest_code ingests code into PageTable and ActiveCanvas."""
    pipeline = CognitivePipeline(
        transducer=mock_code_transducer,
        canvas_capacity=512,
    )

    graph = pipeline.ingest_code(PYTHON_TAX_CODE, language_hint="Python")
    assert isinstance(graph, QuantaGraph)
    assert len(graph.nodes) >= 6

    # Verify nodes stored in PageTable
    stored_count = pipeline.page_table.count_nodes()
    assert stored_count >= len(graph.nodes)

    # Verify ActiveCanvas contains nodes bounded by capacity
    assert len(pipeline.active_canvas) <= pipeline.active_canvas.capacity
    assert len(pipeline.active_canvas) > 0


# ---------------------------------------------------------------------------
# Task 9.6.5: Spreading Activation Sub-Graph Code Retrieval
# ---------------------------------------------------------------------------

def test_spreading_activation_code_retrieval(mock_code_transducer):
    """Verify spreading activation retrieves code callers in < 5ms without distractors."""
    pipeline = CognitivePipeline(
        transducer=mock_code_transducer,
        canvas_capacity=512,
    )

    pipeline.ingest_code(CALLER_CORPUS_CODE, language_hint="Java")

    # Warmup query to avoid initial SQLite connection / module load cold start
    _ = pipeline.retrieve_context(
        query="Who calls processOrder?",
        format="sexpr",
        max_tokens=300,
    )

    t0 = time.perf_counter()
    retrieved_sexpr = pipeline.retrieve_context(
        query="Who calls processOrder?",
        format="sexpr",
        max_tokens=300,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Sub-5ms requirement
    assert latency_ms < 5.0, f"Spreading activation latency was {latency_ms:.2f} ms (expected < 5.0 ms)"

    # Verify relevant callers retrieved
    assert "handleCheckout" in retrieved_sexpr or "OrderService" in retrieved_sexpr
    assert "auditTransaction" in retrieved_sexpr or "PaymentAuditService" in retrieved_sexpr

    # Verify irrelevant distractor excluded
    assert "restockItem" not in retrieved_sexpr
    assert "updateInventory" not in retrieved_sexpr


# ---------------------------------------------------------------------------
# Task 9.6.6: Neural Reverse Code Realization
# ---------------------------------------------------------------------------

def test_neural_reverse_code_realization(mock_code_transducer, compiler):
    """Verify TwoWayTranslationPipeline.realize_code generates valid code from ASG."""
    ext = mock_code_transducer.transduce(text=PYTHON_TAX_CODE)
    graph = compiler.compile(ext, validate=False)

    translator = TwoWayTranslationPipeline(transducer=mock_code_transducer)

    # Realize to Python
    py_code = translator.realize_code(graph, target_lang="python")
    assert "def calculate_tax" in py_code
    assert "get_tax_rate" in py_code
    assert "return subtotal * rate" in py_code

    # Realize to Java
    ext_java = mock_code_transducer.transduce(text=JAVA_PROCESSOR_CODE)
    graph_java = compiler.compile(ext_java, validate=False)
    java_code = translator.realize_code(graph_java, target_lang="java")
    assert "public class OrderProcessor" in java_code
    assert "extends BaseProcessor" in java_code
    assert "implements IProcessor" in java_code


# ---------------------------------------------------------------------------
# Task 9.6.7: Zero Language-Specific Parser Imports Check
# ---------------------------------------------------------------------------

def test_zero_language_specific_parser_imports():
    """Verify live code transduction uses zero language-specific parser libraries."""
    import parser.unsloth_transducer as ut_mod
    import pipeline.cognitive_pipeline as cp_mod

    ut_src = inspect.getsource(ut_mod)
    cp_src = inspect.getsource(cp_mod)

    # Check that live transduction does not import language AST parsers
    prohibited_imports = [
        "tree_sitter",
        "javaparser",
        "javalang",
        "esprima",
        "syn_parser",
        "CodeLanguageAdapter",
    ]

    for p in prohibited_imports:
        assert p not in ut_src, f"Prohibited parser '{p}' found in unsloth_transducer.py"
        assert p not in cp_src, f"Prohibited parser '{p}' found in cognitive_pipeline.py"


# ---------------------------------------------------------------------------
# Task 9.6.8: Attachment Scripts Transduction (Bubble Sort & UNO Tournament Simulator)
# ---------------------------------------------------------------------------

BUBBLE_SORT_SEXPR = """(graph :chunk-id "py_bubble_sort"
  (entity :id E1 :type ARTIFACT :label "bubble_sort" :surface "def bubble_sort(arr: List[T]) -> List[T]:")
  (entity :id E2 :type ARTIFACT :label "arr" :surface "arr")
  (entity :id E3 :type ARTIFACT :label "swapped" :surface "swapped = False")
  (entity :id E4 :type ARTIFACT :label "adjacent_pair" :surface "arr[j], arr[j + 1]")
  (event :id Ev1 :pred iterate :agent E1 :patient E2 :time "outer loop" :tense PRESENT :polarity TRUE :raw-text "The bubble_sort function iterates through the list arr using an outer loop for i in range(n).")
  (event :id Ev2 :pred compare :agent E1 :patient E4 :time "inner loop" :tense PRESENT :polarity TRUE :raw-text "The inner loop compares adjacent elements arr[j] and arr[j + 1] and swaps them if out of ascending order.")
  (event :id Ev3 :pred terminate :agent E1 :patient E3 :time "early break" :tense PRESENT :polarity TRUE :raw-text "The function checks the swapped flag and breaks early if no swaps occurred, guaranteeing O(n) best-case complexity.")
  (relation :type CFG_NEXT :source Ev1 :target Ev2)
  (relation :type CFG_NEXT :source Ev2 :target Ev3)
)"""

UNO_SIMULATOR_SEXPR = """(graph :chunk-id "py_uno_simulator"
  (entity :id E1 :type ARTIFACT :label "Card" :surface "class Card")
  (entity :id E2 :type ARTIFACT :label "PlayerStrategy" :surface "class PlayerStrategy")
  (entity :id E3 :type ARTIFACT :label "choose_card" :surface "def choose_card(self, hand, active_color, active_value)")
  (entity :id E4 :type ARTIFACT :label "RandomStrategy" :surface "class RandomStrategy(PlayerStrategy)")
  (entity :id E5 :type ARTIFACT :label "ColorMatchPriorityStrategy" :surface "class ColorMatchPriorityStrategy(PlayerStrategy)")
  (entity :id E6 :type ARTIFACT :label "UnoGameSimulator" :surface "class UnoGameSimulator")
  (entity :id E7 :type ARTIFACT :label "run_tournament" :surface "def run_tournament(num_games: int = 1000)")
  (entity :id E8 :type ARTIFACT :label "print_stats" :surface "def print_stats(stats)")
  (event :id Ev1 :pred define_interface :agent E2 :patient E3 :tense PRESENT :polarity TRUE :raw-text "PlayerStrategy defines choose_card taking hand, active_color, and active_value, returning a valid card tuple or None to draw.")
  (event :id Ev2 :pred inherit :agent E4 :patient E2 :tense PRESENT :polarity TRUE :raw-text "RandomStrategy inherits from PlayerStrategy and plays a randomly chosen valid card from hand.")
  (event :id Ev3 :pred inherit :agent E5 :patient E2 :tense PRESENT :polarity TRUE :raw-text "ColorMatchPriorityStrategy inherits from PlayerStrategy and prioritizes cards matching the active color.")
  (event :id Ev4 :pred simulate :agent E6 :patient E3 :tense PRESENT :polarity TRUE :raw-text "UnoGameSimulator executes 4-player turns invoking choose_card until a player empties their hand.")
  (event :id Ev5 :pred execute_tournament :agent E7 :patient E6 :tense PRESENT :polarity TRUE :raw-text "run_tournament configures 1 ColorMatchPriorityStrategy for Player 0 and 3 RandomStrategy players for 1000 games.")
  (relation :type INHERITS_FROM :source E4 :target E2)
  (relation :type INHERITS_FROM :source E5 :target E2)
  (relation :type CALLS :source E6 :target E3)
  (relation :type CALLS :source E7 :target E6)
)"""


def test_bubble_sort_transduction_and_retrieval():
    """Verify Bubble Sort attachment script transduces to ASG and supports sub-5ms retrieval."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    bs_path = repo_root / "scripts" / "bubble_sort.py"
    assert bs_path.exists(), "scripts/bubble_sort.py must exist as attachment script"
    bs_code = bs_path.read_text(encoding="utf-8")

    mock = MockUnslothTransducer()
    mock.register_fixture(bs_code, BUBBLE_SORT_SEXPR)
    mock.register_fixture("py_bubble_sort", BUBBLE_SORT_SEXPR)

    pipeline = CognitivePipeline(
        transducer=mock,
        canvas_capacity=512,
    )
    graph = pipeline.ingest_code(bs_code, language_hint="Python", chapter_id="py_bubble_sort")
    assert len(graph.nodes) >= 4

    # Verify spreading activation retrieval of bubble sort algorithm context
    ctx = pipeline.retrieve_context("bubble_sort adjacent compare swap early termination swapped", format="english", max_tokens=250)
    assert "bubble_sort" in ctx.lower() or "bubble" in ctx.lower()
    assert "adjacent" in ctx.lower() or "swap" in ctx.lower()


def test_uno_simulator_transduction_and_retrieval():
    """Verify UNO tournament simulator attachment script transduces to ASG with inheritance edges."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    uno_path = repo_root / "scripts" / "uno_simulator.py"
    assert uno_path.exists(), "scripts/uno_simulator.py must exist as attachment script"
    uno_code = uno_path.read_text(encoding="utf-8")

    mock = MockUnslothTransducer()
    mock.register_fixture(uno_code, UNO_SIMULATOR_SEXPR)
    mock.register_fixture("py_uno_simulator", UNO_SIMULATOR_SEXPR)

    pipeline = CognitivePipeline(
        transducer=mock,
        canvas_capacity=512,
    )
    graph = pipeline.ingest_code(uno_code, language_hint="Python", chapter_id="py_uno_simulator")
    assert len(graph.nodes) >= 6

    # Verify inheritance relation edge
    r_strat = [n for n in graph.nodes.values() if n.literal == "RandomStrategy"]
    p_strat = [n for n in graph.nodes.values() if n.literal == "PlayerStrategy"]
    assert len(r_strat) == 1 and len(p_strat) == 1
    assert "INHERITS_FROM" in r_strat[0].edges
    assert p_strat[0].cid in r_strat[0].edges["INHERITS_FROM"]

    # Verify spreading activation retrieval of PlayerStrategy & choose_card
    ctx = pipeline.retrieve_context("PlayerStrategy choose_card active_color active_value", format="english", max_tokens=250)
    assert "playerstrategy" in ctx.lower()
    assert "choose_card" in ctx.lower()

