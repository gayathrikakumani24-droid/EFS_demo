with open('hardware_graphrag/core/agents/spec_classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''class SpecType(str, Enum):
    COMBINATIONAL = "combinational"
    FSM_CONTROLLER = "fsm_controller"
    DATAPATH_PIPELINE = "datapath_pipeline"
    REGISTER_MAP = "register_map"
    FIFO_MEMORY = "fifo_memory"
    PROTOCOL_ENGINE = "protocol_engine"'''

replacement = '''class SpecType(str, Enum):
    COMBINATIONAL = "combinational"
    FSM_CONTROLLER = "fsm_controller"
    DATAPATH_PIPELINE = "datapath_pipeline"
    REGISTER_MAP = "register_map"
    FIFO_MEMORY = "fifo_memory"
    PROTOCOL_ENGINE = "protocol_engine"


class ArchitectureClass(str, Enum):
    COMBINATIONAL = "COMBINATIONAL"
    SEQUENTIAL = "SEQUENTIAL"
    FSM_CONTROLLER = "FSM_CONTROLLER"
    DATAPATH = "DATAPATH"
    MEMORY_INTERFACE = "MEMORY_INTERFACE"
    BUS_INTERFACE = "BUS_INTERFACE"
    ARITHMETIC_PROCESSING_UNIT = "ARITHMETIC_PROCESSING_UNIT"
    PROTOCOL_ENGINE = "PROTOCOL_ENGINE"


from dataclasses import dataclass, field


@dataclass
class ArchitectureClassificationResult:
    primary_class: ArchitectureClass
    secondary_traits: List[str] = field(default_factory=list)
    has_clock: bool = False
    has_reset: bool = False
    requires_fsm: bool = False
    reasoning: str = ""


def classify_hardware_architecture(
    efs_ir: Optional[Any] = None,
    spec_text: str = "",
    **kwargs
) -> ArchitectureClassificationResult:
    """Classifies design architecture deterministically without guessing."""
    text_lower = (spec_text or "").lower()

    # 1. Clock and reset detection
    has_clock = False
    has_reset = False
    port_names: Set[str] = set()
    if efs_ir and hasattr(efs_ir, "signals"):
        for s in efs_ir.signals:
            s_name = (getattr(s, "name", "") or "").lower()
            port_names.add(s_name)
            if getattr(s, "semantic_role", "") == "clock" or s_name in ("clk", "clock", "aclk"):
                has_clock = True
            if getattr(s, "semantic_role", "") == "reset" or any(k in s_name for k in ("rst", "reset", "aresetn")):
                has_reset = True

    if not has_clock:
        has_clock = bool(re.search(r"\\b(?:clk|clock|posedge|aclk)\\b", text_lower))
    if not has_reset:
        has_reset = bool(re.search(r"\\b(?:rst_n|reset|rst|aresetn)\\b", text_lower))

    # Instructions check
    has_instructions = bool(
        (efs_ir and (getattr(efs_ir, "opcodes", None) or getattr(efs_ir, "instructions", None))) or
        re.search(r"\\b(?:opcode|instruction\\s+decoder|instruction\\s+set|instruction\\s+format|mnemonics?)\\b", text_lower)
    )

    traits: List[str] = []
    if any(k in text_lower for k in ("arithmetic", "multiplier", "accumulator", "matrix", "alu", "multiply", "accumulate")):
        traits.append("ARITHMETIC")

    # Pure combinational check
    if not has_clock and not has_reset and not any(k in text_lower for k in ("state", "fsm", "clock", "seq")):
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.COMBINATIONAL,
            secondary_traits=traits,
            has_clock=False,
            has_reset=False,
            requires_fsm=False,
            reasoning="No clock or reset signals detected; classified as pure combinational."
        )

    # Explicit FSM controller check
    has_explicit_fsm = bool(
        (efs_ir and getattr(efs_ir, "fsms", None)) or
        re.search(r"\\b(?:states?[:\\s]+[A-Z0-9_]+|sequence\\s+controller|state\\s+machine)\\b", spec_text) or
        ("->" in spec_text and "state" in text_lower)
    )

    # Arithmetic processing unit
    if traits and ("ARITHMETIC" in traits) and (has_instructions or "matrix" in text_lower or "multiplier" in text_lower or "accumulator" in text_lower):
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.ARITHMETIC_PROCESSING_UNIT,
            secondary_traits=traits,
            has_clock=has_clock,
            has_reset=has_reset,
            requires_fsm=has_explicit_fsm,
            reasoning="Arithmetic processing operations detected."
        )

    # FIFO / Memory
    if not has_instructions and ("fifo" in text_lower or "circular buffer" in text_lower or "write_pointer" in text_lower):
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.MEMORY_INTERFACE,
            secondary_traits=traits,
            has_clock=has_clock,
            has_reset=has_reset,
            requires_fsm=False,
            reasoning="Memory / FIFO buffer detected."
        )

    # Bus / CSR Register Map
    if any(k in text_lower for k in ("reg_addr", "register bank", "csr", "offset 0x")):
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.BUS_INTERFACE,
            secondary_traits=traits,
            has_clock=has_clock,
            has_reset=has_reset,
            requires_fsm=False,
            reasoning="Register bank / bus interface detected."
        )

    # Explicit FSM controller
    if has_explicit_fsm:
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.FSM_CONTROLLER,
            secondary_traits=traits,
            has_clock=has_clock,
            has_reset=has_reset,
            requires_fsm=True,
            reasoning="Explicit sequence controller / state machine detected."
        )

    # Pipeline datapath
    if "pipeline" in text_lower or "stage_valid" in text_lower or "stage_data" in text_lower:
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.DATAPATH,
            secondary_traits=traits,
            has_clock=has_clock,
            has_reset=has_reset,
            requires_fsm=False,
            reasoning="Pipelined datapath detected."
        )

    # Counter / sequential datapath
    if "counter" in text_lower or "increments" in text_lower:
        return ArchitectureClassificationResult(
            primary_class=ArchitectureClass.DATAPATH,
            secondary_traits=traits,
            has_clock=has_clock,
            has_reset=has_reset,
            requires_fsm=False,
            reasoning="Sequential counter datapath detected."
        )

    return ArchitectureClassificationResult(
        primary_class=ArchitectureClass.SEQUENTIAL,
        secondary_traits=traits,
        has_clock=has_clock,
        has_reset=has_reset,
        requires_fsm=False,
        reasoning="Sequential hardware logic detected."
    )'''

normalized_content = content.replace('\r\n', '\n')
if target in normalized_content:
    new_content = normalized_content.replace(target, replacement, 1)
    with open('hardware_graphrag/core/agents/spec_classifier.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS")
else:
    print("TARGET NOT FOUND")
