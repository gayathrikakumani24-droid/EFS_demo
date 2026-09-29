"""
End-to-End Tests for Generation Context Contract and Context-Compilation-to-Generation Pipeline.

Verifies:
1. GenerationContext contract is specification-agnostic and enforces clean boundaries.
2. Context Compiler output is directly used by final generation prompts without raw re-expansion.
3. original_context != compiled_context.
4. final_llm_input contains compiled_context while NOT containing the raw DocumentIR / full spec.
5. All agents (planning_agent, spec_agent, rtl_agent, plantuml_agent) transmit strictly bounded prompts (< 4,000 - 8,000 tokens).
6. Semantic constructs (opcodes, instructions, registers, FSM states/transitions, ports) are preserved.
7. Telemetry is emitted with reduction ratios and token measurements.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from core.context_compiler.generation_context import GenerationContext
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSSignal, EFSRegister, EFSFSM,
    EFSState, EFSTransition, EFSOpcode, EFSInstruction,
    EFSTransaction, EFSConstraint
)
from core.requirement_ir.models import (
    RequirementIR, DocumentIR, DocumentBlock, DiscoveredEntity, AtomicRequirement
)


def _build_large_mock_requirement_ir(num_blocks: int = 30) -> RequirementIR:
    """Construct a large mock RequirementIR that would normally expand to 50,000+ tokens."""
    blocks = []
    entities = []
    reqs = []

    # Large text blocks (simulating 50+ page specification text)
    for i in range(num_blocks):
        large_text = f"Chapter {i}: Detailed hardware specifications and descriptions. " * 80
        blocks.append(DocumentBlock(
            block_id=f"BLK_{i:03d}",
            type="paragraph",
            text=large_text,
            page=i + 1
        ))

    doc_ir = DocumentIR(
        document_id="DOC_TEST_LARGE",
        document_hash="HASH_12345",
        metadata={"summary": "A high-performance Matrix Coprocessor unit with AXI4 streaming interface."},
        blocks=blocks
    )

    # Specific structured hardware entities
    entities.extend([
        DiscoveredEntity(name="clk", type="clock", attributes={"direction": "input", "width": "1", "description": "System clock"}),
        DiscoveredEntity(name="rst_n", type="reset", attributes={"direction": "input", "width": "1", "description": "Active-low reset"}),
        DiscoveredEntity(name="m_axis_tdata", type="signal", attributes={"direction": "output", "width": "64", "description": "Output data bus"}),
        DiscoveredEntity(name="m_axis_tvalid", type="signal", attributes={"direction": "output", "width": "1", "description": "Output valid handshake"}),
        DiscoveredEntity(name="m_axis_tready", type="signal", attributes={"direction": "input", "width": "1", "description": "Downstream ready"}),
        DiscoveredEntity(name="CTRL_REG", type="register", attributes={"offset": "0x00", "access": "RW", "description": "Control register"}),
        DiscoveredEntity(name="STATUS_REG", type="register", attributes={"offset": "0x04", "access": "RO", "description": "Status register"}),
        DiscoveredEntity(name="BURST_LEN", type="parameter", attributes={"value": "16", "description": "Burst transfer length"}),
        DiscoveredEntity(name="MADD", type="instruction", description="Matrix Multiply-Add instruction"),
        DiscoveredEntity(name="MADD_OP", type="opcode", attributes={"value": "0x1F", "encoding": "0x1F", "description": "MADD opcode encoding"}),
    ])

    # Specific functional requirements & FSM transitions
    reqs.extend([
        AtomicRequirement(
            statement="System transitions from IDLE to COMPUTE when start signal is asserted.",
            state="IDLE",
            next_state="COMPUTE",
            condition="start == 1'b1",
            action="busy <= 1'b1",
            type="BEHAVIOR"
        ),
        AtomicRequirement(
            statement="System transitions from COMPUTE to DONE when matrix multiplication completes.",
            state="COMPUTE",
            next_state="DONE",
            condition="matrix_done == 1'b1",
            action="done <= 1'b1",
            type="BEHAVIOR"
        ),
        AtomicRequirement(
            statement="System transitions from DONE to IDLE when ack is received.",
            state="DONE",
            next_state="IDLE",
            condition="ack == 1'b1",
            action="done <= 1'b0",
            type="BEHAVIOR"
        ),
        AtomicRequirement(
            statement="Max clock frequency must be 500MHz.",
            type="CONSTRAINT"
        )
    ])

    return RequirementIR(
        doc_ir=doc_ir,
        document_id="DOC_TEST_LARGE",
        entities=entities,
        requirements=reqs
    )


def _build_mock_efs_ir() -> EFSIR:
    """Build a mock EFS IR with components, signals, registers, FSM, and opcodes."""
    comp = EFSComponent(
        component_id="COMP_MAT",
        name="Matrix_Engine",
        type="accelerator",
        description="Matrix computation engine"
    )
    signals = [
        EFSSignal(signal_id="s1", name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(signal_id="s2", name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(signal_id="s3", name="start", direction="input", width="1", semantic_role="control"),
        EFSSignal(signal_id="s4", name="busy", direction="output", width="1", semantic_role="status"),
        EFSSignal(signal_id="s5", name="done", direction="output", width="1", semantic_role="status"),
    ]
    regs = [
        EFSRegister(register_id="r1", name="CFG_REG", offset="0x0", width=32, description="Configuration")
    ]
    fsm = EFSFSM(
        fsm_id="f1",
        name="Engine_FSM",
        states=[EFSState(name="IDLE"), EFSState(name="RUN"), EFSState(name="DONE")],
        initial_state="IDLE",
        transitions=[
            EFSTransition(source_state="IDLE", target_state="RUN", condition="start"),
            EFSTransition(source_state="RUN", target_state="DONE", condition="complete"),
            EFSTransition(source_state="DONE", target_state="IDLE", condition="ack"),
        ]
    )
    opcodes = [
        EFSOpcode(opcode_id="op1", mnemonic="MAT_MUL", binary_encoding="0x3A")
    ]
    instructions = [
        EFSInstruction(instruction_id="in1", name="MAT_MUL", mnemonic="MAT_MUL", description="Matrix Multiply")
    ]
    return EFSIR(
        components=[comp],
        signals=signals,
        registers=regs,
        fsms=[fsm],
        opcodes=opcodes,
        instructions=instructions
    )


class TestGenerationContextContract:
    """Tests for the explicit GenerationContext contract boundary."""

    def test_generation_context_from_requirement_model(self):
        req_ir = _build_large_mock_requirement_ir(num_blocks=25)
        raw_dict = req_ir.to_full_combined_dict()
        raw_dump_len = len(json.dumps(raw_dict))

        # Compile into GenerationContext
        gen_ctx = GenerationContext.from_requirement_model(req_ir, artifact_type="planning")
        compiled_json = gen_ctx.to_prompt_json()

        # 1. Verify original_context != compiled_context
        assert gen_ctx.original_context_tokens > 0
        assert gen_ctx.compiled_context_tokens > 0
        assert gen_ctx.original_context_tokens > gen_ctx.compiled_context_tokens

        # 2. Verify compiled context is non-empty
        assert len(gen_ctx.signals) > 0
        assert len(gen_ctx.registers) > 0
        assert len(gen_ctx.fsm.get("states", [])) == 3
        assert len(gen_ctx.opcodes) > 0
        assert len(gen_ctx.parameters) > 0

        # 3. Verify that raw text blocks from DocumentIR are NOT present
        assert "Detailed hardware specifications and descriptions" not in compiled_json
        assert "BLK_020" not in compiled_json

        # 4. Verify dramatic token reduction (> 80%)
        reduction = (gen_ctx.original_context_tokens - gen_ctx.compiled_context_tokens) / gen_ctx.original_context_tokens
        assert reduction > 0.80

    def test_generation_context_from_spec_inputs(self):
        req_ir = _build_large_mock_requirement_ir(num_blocks=25)
        plan = {
            "system_architecture": "Pipelined 2D Systolic Array",
            "submodules": [{"name": "PE_Array", "purpose": "Processing elements"}],
            "clock_domains": [{"name": "clk", "reset_strategy": "sync active-low"}],
            "interface_mapping": {"external_bus": "AXI4", "internal_bus": "RegBus"}
        }
        protocol_rules = [
            {"citation": "AXI4 §3.1", "text": "A value of x on valid is prohibited when reset is deasserted."},
            {"citation": "AXI4 §3.2", "text": "TREADY may be asserted before, after, or coincident with TVALID."}
        ]

        gen_ctx = GenerationContext.from_spec_inputs(req_ir, plan, protocol_rules, artifact_type="design_spec")
        compiled_json = gen_ctx.to_prompt_json()

        # Verify structured elements
        assert "PE_Array" in compiled_json
        assert "AXI4" in compiled_json
        assert "3.1" in compiled_json
        assert "Pipelined 2D Systolic Array" in compiled_json
        assert len(gen_ctx.signals) > 0
        assert len(gen_ctx.registers) > 0

        # Verify raw DocumentIR text was excluded
        assert "Detailed hardware specifications and descriptions" not in compiled_json
        assert gen_ctx.compiled_context_tokens < 3500


class TestPipelinePromptWiring:
    """Tests verifying that generation agents use ONLY compiled context and do not re-expand."""

    @patch("core.agents.planning_agent.get_llm_client")
    def test_planning_agent_uses_compiled_context_strictly(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.available = True
        mock_llm.last_usage = {"prompt_tokens": 1200, "completion_tokens": 450, "total_tokens": 1650}
        mock_llm.complete_json.return_value = {
            "system_architecture": "Systolic Array",
            "submodules": [{"name": "PE", "purpose": "compute", "interfaces": []}],
            "clock_domains": [{"name": "clk", "reset_strategy": "active-low", "description": "core"}],
            "interface_mapping": {"external_bus": "AXI4", "internal_bus": "Local", "description": "bridge"},
            "fsm_architecture": {"states": [{"name": "IDLE", "actions": "none"}], "description": "fsm"},
            "registers_summary": "regs",
            "memories_summary": "mems"
        }
        mock_get_llm.return_value = mock_llm

        from core.agents.planning_agent import create_plan
        req_ir = _build_large_mock_requirement_ir(num_blocks=30)

        plan = create_plan(req_ir)
        assert plan is not None
        assert mock_llm.complete_json.called

        # Inspect prompt passed to LLM
        call_args = mock_llm.complete_json.call_args[0]
        sys_prompt, user_prompt = call_args[0], call_args[1]

        # 1. Prompt must contain compiled context boundary
        assert "=== COMPILED REQUIREMENT CONTEXT ===" in user_prompt
        # 2. Prompt must NOT contain raw DocumentIR dumps
        assert "Detailed hardware specifications and descriptions" not in user_prompt
        assert "BLK_029" not in user_prompt
        # 3. Prompt token estimate must be compact (< 4000 tokens)
        assert len(user_prompt) // 4 < 4000

    @patch("core.agents.spec_agent.get_llm_client")
    def test_spec_agent_uses_compiled_context_strictly(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.available = True
        mock_llm.last_usage = {"prompt_tokens": 1800, "completion_tokens": 1200, "total_tokens": 3000}
        mock_llm.complete_text.return_value = "# Hardware Specification\nDetailed synthesizable design."
        mock_get_llm.return_value = mock_llm

        from core.agents.spec_agent import generate_design_spec
        req_ir = _build_large_mock_requirement_ir(num_blocks=30)
        plan = {"system_architecture": "Systolic Array", "submodules": []}
        protocol_rules = [{"citation": "AXI §1", "text": "Valid must not depend on Ready."}]

        spec = generate_design_spec(req_ir, plan, protocol_rules)
        assert spec is not None
        assert mock_llm.complete_text.called

        call_args = mock_llm.complete_text.call_args[0]
        sys_prompt, user_prompt = call_args[0], call_args[1]

        # 1. Prompt must use compiled boundary
        assert "=== COMPILED DESIGN SPECIFICATION CONTEXT ===" in user_prompt
        # 2. Prompt must NOT re-expand raw DocumentIR text
        assert "Detailed hardware specifications and descriptions" not in user_prompt
        # 3. Size must be bounded
        assert len(user_prompt) // 4 < 4000

    @patch("core.agents.rtl_agent.get_llm_client")
    def test_rtl_agent_stage1_does_not_reinject_raw_spec(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.available = True
        mock_llm.last_usage = {"prompt_tokens": 2200, "completion_tokens": 800, "total_tokens": 3000}
        mock_llm.complete_json.return_value = {
            "module_name": "Matrix_Engine",
            "specification_status": {"status": "COMPLETE", "blocking_issues": []},
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "start", "direction": "input", "width": "1"},
                {"name": "busy", "direction": "output", "width": "1"},
                {"name": "done", "direction": "output", "width": "1"}
            ],
            "fsm": {
                "states": ["IDLE", "RUN", "DONE"],
                "initial_state": "IDLE",
                "transitions": []
            },
            "datapath_strategy": "Registered compute",
            "reset_polarity": "active_low"
        }
        mock_get_llm.return_value = mock_llm

        from core.agents.rtl_agent import plan_rtl
        efs_ir = _build_mock_efs_ir()
        massive_raw_design_spec = "=== PROPRIETARY SPECIFICATION DUMP ===\n" + ("Raw design details text block. " * 500)

        plan = plan_rtl(massive_raw_design_spec, target="Matrix_Engine", efs_ir=efs_ir)
        assert plan is not None
        assert mock_llm.complete_json.called

        call_args = mock_llm.complete_json.call_args[0]
        plan_prompt = call_args[1]

        # 1. Prompt must use compiled boundary
        assert "=== COMPILED GENERATION CONTEXT ===" in plan_prompt
        # 2. Must NOT re-inject the raw specification dump
        assert "=== SPECIFICATION ===" not in plan_prompt
        assert "PROPRIETARY SPECIFICATION DUMP" not in plan_prompt
        # 3. Context size must be strictly bounded (< 8000 tokens)
        assert len(plan_prompt) // 4 < 8000
        # 4. Critical semantic constructs (opcodes, registers, FSM) must be present in compiled context
        assert "MAT_MUL" in plan_prompt
        assert "CFG_REG" in plan_prompt

    @patch("core.agents.plantuml_agent.get_llm_client")
    def test_plantuml_agent_prompt_is_bounded_and_grounded(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.available = True
        mock_llm.last_usage = {"prompt_tokens": 1500, "completion_tokens": 400, "total_tokens": 1900}
        mock_llm.complete_json.return_value = {
            "architecture": "@startuml\n[Matrix_Engine]\n@enduml",
            "fsm": "@startuml\n[*] --> IDLE\n@enduml",
            "sequence": "@startuml\nHost -> Matrix_Engine: start\n@enduml"
        }
        mock_get_llm.return_value = mock_llm

        from core.agents.plantuml_agent import generate_diagrams
        efs_ir = _build_mock_efs_ir()
        plan = {"system_architecture": "Engine plan"}
        massive_spec = "Proprietary specification prose. " * 600

        diagrams = generate_diagrams(massive_spec, plan, efs_ir=efs_ir)
        assert diagrams is not None
        assert mock_llm.complete_json.called

        call_args = mock_llm.complete_json.call_args[0]
        puml_prompt = call_args[1]

        # 1. Compiled diagram context header must be present
        assert "=== COMPILED HARDWARE DIAGRAM CONTEXT ===" in puml_prompt
        # 2. Raw specification prose must NOT be dumped into the diagram prompt
        assert "Proprietary specification prose." not in puml_prompt
        # 3. Tokens must be bounded
        assert len(puml_prompt) // 4 < 4000
