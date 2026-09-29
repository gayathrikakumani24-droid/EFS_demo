"""
Comprehensive Test Suite for Artifact-Specific Context Compilation Layer.

Tests:
A. Common resolver (ID -> EFS object, ID -> canonical human-readable name)
B. Diagram context compiler (Architecture, FSM, Sequence)
C. RTL context compiler (target module, ports, signals, registers, clock/reset, FSM, datapath, protocol rules)
D. Grounding validation (reject nonexistent/invented hardware, detect leaked internal IDs)
E. Deduplication (remove duplicate entities, repeated signals, redundant protocol rules)
F. Token budgeting (priority-based pruning, no arbitrary string truncation, preserves valid JSON)
G. Telemetry & Metrics (observability, token reduction ratio)
"""

import json
import pytest
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSInterface,
    EFSSignal,
    EFSRegister,
    EFSRegisterField,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSFunctionalBehavior,
    EFSTimingRule,
    EFSProtocolRule
)
from core.context_compiler import (
    compile_context,
    ContextResolver,
    GroundingValidator,
    TokenBudgetController,
    CompilationMetrics,
    estimate_tokens,
    DiagramContextCompiler,
    RTLContextCompiler,
    TBContextCompiler,
    UVMContextCompiler,
    SVAContextCompiler
)


@pytest.fixture
def sample_soc_ir():
    """Create a structured multi-module SoC EFS IR with realistic components."""
    # Components
    ctrl = EFSComponent(
        component_id="COMP_001",
        name="AXI_DMA_Controller",
        type="controller",
        description="High throughput DMA controller with AXI master interface",
        clock_domain="clk_sys",
        reset_domain="rst_sys_n",
        interfaces=["IF_AXI_M", "IF_CFG"]
    )
    fifo = EFSComponent(
        component_id="COMP_002",
        name="Data_Buffer_FIFO",
        type="fifo",
        description="Internal FIFO buffer between DMA and memory",
        clock_domain="clk_sys",
        reset_domain="rst_sys_n",
        interfaces=["IF_FIFO_IN", "IF_FIFO_OUT"]
    )

    # Interfaces
    if_axi = EFSInterface(
        interface_id="IF_AXI_M",
        name="m_axi",
        protocol="AXI4",
        role="master",
        source_component="COMP_001",
        signals=["SIG_AWVALID", "SIG_AWREADY", "SIG_WDATA"]
    )
    if_fifo = EFSInterface(
        interface_id="IF_FIFO_IN",
        name="fifo_in",
        protocol="Native",
        role="slave",
        source_component="COMP_002",
        signals=["SIG_FIFO_DIN", "SIG_FIFO_WR"]
    )

    # Signals
    clk = EFSSignal(signal_id="SIG_CLK", name="clk_sys", direction="input", semantic_role="clock", owner="COMP_001")
    rst = EFSSignal(signal_id="SIG_RST", name="rst_sys_n", direction="input", semantic_role="reset", owner="COMP_001")
    awvalid = EFSSignal(signal_id="SIG_AWVALID", name="m_axi_awvalid", direction="output", width="1", semantic_role="control", owner="COMP_001")
    awready = EFSSignal(signal_id="SIG_AWREADY", name="m_axi_awready", direction="input", width="1", semantic_role="status", owner="COMP_001")
    wdata = EFSSignal(signal_id="SIG_WDATA", name="m_axi_wdata", direction="output", width="64", semantic_role="data", owner="COMP_001")
    
    # FIFO signals (should not be in DMA RTL context)
    fifo_din = EFSSignal(signal_id="SIG_FIFO_DIN", name="fifo_din", direction="input", width="64", semantic_role="data", owner="COMP_002")
    fifo_wr = EFSSignal(signal_id="SIG_FIFO_WR", name="fifo_wr_en", direction="input", width="1", semantic_role="control", owner="COMP_002")

    # Registers
    reg_ctrl = EFSRegister(
        register_id="REG_001",
        name="DMA_CONTROL",
        offset="0x00",
        width=32,
        owner="COMP_001",
        description="Main DMA start and config register",
        fields=[
            EFSRegisterField(name="START", msb=0, lsb=0, description="Start transfer"),
            EFSRegisterField(name="BURST_LEN", msb=7, lsb=4, description="Burst transfer length")
        ]
    )
    reg_fifo_stat = EFSRegister(
        register_id="REG_002",
        name="FIFO_STATUS",
        offset="0x10",
        width=32,
        owner="COMP_002",
        description="FIFO status flags"
    )

    # FSM for DMA Controller
    fsm_dma = EFSFSM(
        fsm_id="FSM_001",
        name="DMA_Engine_FSM",
        initial_state="IDLE",
        states=[EFSState(name="IDLE"), EFSState(name="ADDR_PHASE"), EFSState(name="DATA_BURST"), EFSState(name="DONE")],
        transitions=[
            EFSTransition(source_state="IDLE", target_state="ADDR_PHASE", condition="DMA_CONTROL[0] == 1'b1"),
            EFSTransition(source_state="ADDR_PHASE", target_state="DATA_BURST", condition="m_axi_awready == 1'b1"),
            EFSTransition(source_state="DATA_BURST", target_state="DONE", condition="burst_count == 4'h8"),
            EFSTransition(source_state="DONE", target_state="IDLE", condition="1'b1")
        ]
    )

    # Functional Behavior / Datapath
    fb_add = EFSFunctionalBehavior(
        behavior_id="FB_001",
        name="burst_counter_increment",
        subject="AXI_DMA_Controller",
        description="Increment burst count on every successful data beat",
        action="burst_count <= burst_count + 1"
    )

    # Protocol Rules
    rule_axi = EFSProtocolRule(
        rule_id="RULE_001",
        name="AXI_AWVALID_AWREADY_HANDSHAKE",
        type="handshake",
        description="Once AWVALID is asserted, it must remain high until AWREADY is high."
    )
    rule_pcie = EFSProtocolRule(
        rule_id="RULE_002",
        name="PCIE_TLP_HEADER",
        type="tlp",
        description="PCIe TLP Header must contain 3DW or 4DW prefix."
    )

    return EFSIR(
        metadata=EFSMetadata(design_name="SoC_Top"),
        components=[ctrl, fifo],
        interfaces=[if_axi, if_fifo],
        signals=[clk, rst, awvalid, awready, wdata, fifo_din, fifo_wr],
        registers=[reg_ctrl, reg_fifo_stat],
        fsms=[fsm_dma],
        functional_behavior=[fb_add],
        protocol_rules=[rule_axi, rule_pcie]
    )


# ---------------------------------------------------------------------------
# A. Common Resolver Tests
# ---------------------------------------------------------------------------

def test_resolver_id_to_canonical_human_names(sample_soc_ir):
    """Test that ContextResolver translates all internal IDs to human-readable names."""
    resolver = ContextResolver(sample_soc_ir)
    
    # Check signal ID resolution
    assert resolver.resolve_signal_name("SIG_AWVALID") == "m_axi_awvalid"
    assert resolver.resolve_signal_name("SIG_CLK") == "clk_sys"
    assert resolver.resolve_signal_name("SIG_FIFO_DIN") == "fifo_din"

    # Check component ID resolution
    assert resolver.resolve_component_name("COMP_001") == "AXI_DMA_Controller"
    assert resolver.resolve_component_name("COMP_002") == "Data_Buffer_FIFO"

    # Check text scrubbing
    raw_expr = "SIG_AWVALID && SIG_AWREADY"
    scrubbed = resolver.scrub_internal_ids(raw_expr)
    assert "SIG_AWVALID" not in scrubbed
    assert "SIG_AWREADY" not in scrubbed
    assert "m_axi_awvalid" in scrubbed
    assert "m_axi_awready" in scrubbed


def test_resolver_never_invents_names():
    """Verify that unknown references are not fabricated into arbitrary signals."""
    ir = EFSIR(signals=[EFSSignal(signal_id="SIG_001", name="real_signal")])
    resolver = ContextResolver(ir)

    # Unknown signal should return unchanged or fallback, not invented hardware
    assert resolver.resolve_signal_name("SIG_UNKNOWN") == "SIG_UNKNOWN"
    assert resolver.resolve_component_name("COMP_NONEXISTENT") == "COMP_NONEXISTENT"


# ---------------------------------------------------------------------------
# B. Diagram Context Compiler Tests
# ---------------------------------------------------------------------------

def test_diagram_context_compilation(sample_soc_ir):
    """Test that DiagramContextCompiler compiles compact architecture, FSM, and sequence contexts."""
    ctx = compile_context("diagram", efs_ir=sample_soc_ir)
    
    assert "architecture" in ctx
    assert "fsm" in ctx
    assert "sequence" in ctx

    # Architecture checks
    arch = ctx["architecture"]
    comp_names = [c["name"] for c in arch["components"]]
    assert "AXI_DMA_Controller" in comp_names
    assert "Data_Buffer_FIFO" in comp_names

    # FSM checks
    fsm = ctx["fsm"]
    assert fsm["initial_state"] == "IDLE"
    assert "ADDR_PHASE" in fsm["states"]
    assert len(fsm["transitions"]) == 4

    # Sequence checks
    seq = ctx["sequence"]
    participants = seq["participants"]
    assert "AXI_DMA_Controller" in participants
    # No internal IDs in sequence participants or signal lists
    assert not any(p.startswith("COMP_") for p in participants)
    assert not any(s.startswith("SIG_") for s in seq["signals"])


def test_diagram_context_excludes_unrelated_raw_payloads(sample_soc_ir):
    """Verify diagram context does not leak raw documents or ungrounded data."""
    raw_spec = "# Massive Document\n" + ("Paragraph text here...\n" * 500)
    ctx = compile_context("diagram", efs_ir=sample_soc_ir, design_spec=raw_spec)
    
    dumped = json.dumps(ctx)
    assert "Massive Document" not in dumped
    assert "Paragraph text here" not in dumped
    assert ctx["_metrics"]["context_reduction_ratio"] != "0.0%"


# ---------------------------------------------------------------------------
# C. RTL Context Compiler Tests
# ---------------------------------------------------------------------------

def test_rtl_context_scopes_to_target_module(sample_soc_ir):
    """Test that RTLContextCompiler scopes context strictly to the target component."""
    ctx = compile_context("rtl", efs_ir=sample_soc_ir, target_component="AXI_DMA_Controller")
    
    assert ctx["module"]["name"] == "AXI_DMA_Controller"

    # Ports & Signals should contain DMA signals, NOT FIFO internal signals
    port_names = [p["name"] for p in ctx["ports"]]
    assert "m_axi_awvalid" in port_names
    assert "m_axi_wdata" in port_names
    assert "clk_sys" in port_names
    assert "rst_sys_n" in port_names
    assert "fifo_din" not in port_names
    assert "fifo_wr_en" not in port_names

    # Registers should contain DMA register, NOT FIFO status register
    reg_names = [r["name"] for r in ctx["registers"]]
    assert "DMA_CONTROL" in reg_names
    assert "FIFO_STATUS" not in reg_names


def test_rtl_context_preserves_behavioral_semantics(sample_soc_ir):
    """Critical Grounding Test: Token reduction MUST NOT remove required behavioral semantics."""
    ctx = compile_context("rtl", efs_ir=sample_soc_ir, target_component="AXI_DMA_Controller")

    # Clock & Reset details preserved
    assert len(ctx["clock_reset"]) >= 2
    rst_info = next(cr for cr in ctx["clock_reset"] if cr["type"] == "reset")
    assert rst_info["polarity"] == "active_low"

    # Register fields preserved
    ctrl_reg = next(r for r in ctx["registers"] if r["name"] == "DMA_CONTROL")
    field_names = [f["name"] for f in ctrl_reg["fields"]]
    assert "START" in field_names
    assert "BURST_LEN" in field_names

    # FSM transitions and conditions preserved
    fsm = ctx["fsm"]
    assert len(fsm["transitions"]) == 4
    trans_0 = fsm["transitions"][0]
    assert trans_0["from"] == "IDLE"
    assert trans_0["to"] == "ADDR_PHASE"
    assert "DMA_CONTROL[0]" in trans_0["condition"]["expression"]

    # Datapath operation preserved
    assert len(ctx["datapath"]) >= 1
    assert "burst_count <= burst_count + 1" in ctx["datapath"][0]["expression"]

    # Protocol rules sliced strictly to AXI (PCIe excluded)
    rule_texts = [r["text"] for r in ctx["protocol_rules"]]
    assert any("AWVALID" in rt for rt in rule_texts)
    assert not any("PCIe TLP" in rt for rt in rule_texts)


# ---------------------------------------------------------------------------
# D. Grounding Validation Tests
# ---------------------------------------------------------------------------

def test_grounding_validator_detects_invented_hardware(sample_soc_ir):
    """Test GroundingValidator rejects invented modules, ports, and internal IDs."""
    fake_context = {
        "module": {"name": "Invented_Crypto_Engine"},
        "ports": [
            {"name": "clk_sys"},
            {"name": "SIG_12345678"},  # Leaked ID
            {"name": "invented_quantum_key"}  # Invented port
        ],
        "registers": [
            {"name": "HALLUCINATED_REG"}
        ],
        "fsm": {"states": ["IDLE", "UNGROUNDED_STATE"], "transitions": []}
    }

    result = GroundingValidator.validate_rtl_context(fake_context, sample_soc_ir)
    assert not result.is_valid
    categories = [v["category"] for v in result.violations]
    assert "MODULE" in categories
    assert "INTERNAL_ID" in categories
    assert "PORT" in categories
    assert "REGISTER" in categories


# ---------------------------------------------------------------------------
# E. Deduplication Tests
# ---------------------------------------------------------------------------

def test_deduplication():
    """Verify deduplication removes redundant signals, ports, and protocol rules."""
    items = [
        {"name": "sig_a", "width": "1"},
        {"name": "sig_a", "width": "1"},  # duplicate
        {"name": "sig_b", "width": "32"}
    ]
    deduped = ContextResolver.deduplicate_list_of_dicts(items, key_fields=["name"])
    assert len(deduped) == 2
    assert [d["name"] for d in deduped] == ["sig_a", "sig_b"]


# ---------------------------------------------------------------------------
# F. Token Budgeting Tests
# ---------------------------------------------------------------------------

def test_budget_controller_priority_pruning():
    """Test that TokenBudgetController prunes P2 explanatory text first and never breaks JSON."""
    oversized_context = {
        "module": {"name": "test_core"},
        "ports": [
            {"name": f"port_{i}", "direction": "input", "width": "32", "description": "A" * 120}
            for i in range(15)
        ],
        "registers": [
            {"name": f"reg_{i}", "width": "32", "description": "B" * 120, "fields": []}
            for i in range(10)
        ],
        "source_evidence": [{"text": "Very long evidence text snippet " * 50}],
        "source_refs": [f"REF_{i}" for i in range(50)],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "transitions": [{"from": "IDLE", "to": "RUN", "condition": "1'b1"}]
        }
    }

    initial_tokens = estimate_tokens(oversized_context)
    pruned, exceeded, warning = TokenBudgetController.enforce_rtl_budget(oversized_context, max_tokens=500)
    final_tokens = estimate_tokens(pruned)

    assert final_tokens < initial_tokens
    # Crucially: JSON structure is 100% valid and parseable
    re_dumped = json.dumps(pruned)
    assert json.loads(re_dumped)
    # Ports and FSM transitions are preserved
    assert len(pruned["ports"]) == 15
    assert pruned["fsm"]["transitions"][0]["from"] == "IDLE"


# ---------------------------------------------------------------------------
# G. Verification Context Compilers (TB, UVM, SVA)
# ---------------------------------------------------------------------------

def test_verification_compilers(sample_soc_ir):
    """Test TB, UVM, and SVA compilers."""
    tb_ctx = compile_context("tb", efs_ir=sample_soc_ir, target_component="AXI_DMA_Controller")
    assert tb_ctx["dut_module"] == "AXI_DMA_Controller"
    assert len(tb_ctx["ports"]) > 0

    uvm_ctx = compile_context("uvm", efs_ir=sample_soc_ir, target_component="AXI_DMA_Controller")
    assert "uvm_components_required" in uvm_ctx
    assert "sequence_item" in uvm_ctx["uvm_components_required"]

    sva_ctx = compile_context("sva", efs_ir=sample_soc_ir, target_component="AXI_DMA_Controller")
    assert "properties_to_assert" in sva_ctx
    assert len(sva_ctx["properties_to_assert"]) > 0
