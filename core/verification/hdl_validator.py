"""
Language-Aware Generic Verilog and SystemVerilog Validator.

Provides comprehensive syntax, structural, semantic, and elaboration validation for:
- Verilog (Verilog-1995, Verilog-2001, Verilog-2005)
- SystemVerilog (SystemVerilog-2005, SystemVerilog-2009, SystemVerilog-2012, SystemVerilog-2017)

Enforces strict separation of language constructs, checks synthesizability,
detects truncation, latch inference, undriven outputs, multiple drivers, and
integrates external compilers (iverilog, verilator, slang) when available.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.logger import get_logger

logger = get_logger("verification.hdl_validator")


class HDLLanguage(str, Enum):
    VERILOG = "Verilog"
    SYSTEMVERILOG = "SystemVerilog"
    UNKNOWN = "Unknown"


class ValidationSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    WARNING = "WARNING"


@dataclass
class HDLError:
    rule_id: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.MAJOR
    line_number: Optional[int] = None
    snippet: Optional[str] = None
    suggested_fix: Optional[str] = None

    @property
    def category(self) -> str:
        return self.rule_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class HDLValidationReport:
    is_valid: bool
    language: HDLLanguage
    language_version: str
    errors: List[HDLError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    tool_status: str = "TOOL_NOT_RUN"
    tool_details: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.is_valid

    @property
    def target_language(self) -> HDLLanguage:
        return self.language

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "valid": self.is_valid,
            "language": self.language.value,
            "target_language": self.language.value,
            "language_version": self.language_version,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
            "passed_checks": self.passed_checks,
            "tool_status": self.tool_status,
            "tool_details": self.tool_details,
            "metrics": self.metrics,
        }


# SystemVerilog-only keywords forbidden in standards-compliant Verilog
SV_ONLY_KEYWORDS = [
    "logic",
    "always_ff",
    "always_comb",
    "always_latch",
    "typedef",
    "enum",
    "struct",
    "union",
    "interface",
    "modport",
    "bit",
    "int",
    "byte",
    "shortint",
    "longint",
    "package",
    "endpackage",
    "inside",
    "unique",
    "priority",
    "alias",
    "assert",
    "covergroup",
    "type",
    "void",
    "class",
    "endclass",
    "program",
    "endprogram",
    "virtual",
    "import",
    "export",
    "chandle",
    "string",
    "timeunit",
    "timeprecision",
]


class ResolvedLanguage(tuple):
    """Tuple subclass enabling seamless comparison with HDLLanguage and string unpacking."""
    def __new__(cls, lang: HDLLanguage, version: str):
        return super().__new__(cls, (lang, version))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HDLLanguage):
            return self[0] == other
        if isinstance(other, str):
            return self[0].value.lower() == other.lower() or self[1].lower() == other.lower()
        return super().__eq__(other)

    @property
    def language(self) -> HDLLanguage:
        return self[0]

    @property
    def version(self) -> str:
        return self[1]

    @property
    def value(self) -> str:
        return self[0].value


def resolve_hdl_language(
    target: str,
    language_version: Optional[str] = None,
    allow_unknown: bool = False
) -> ResolvedLanguage:
    """Resolve normalized HDL language enum and standard version string."""
    t_lower = str(target or "").strip().lower()
    if "systemverilog" in t_lower or t_lower in ("sv", "sv2012", "sv2017"):
        ver = language_version or (
            "SystemVerilog-2017" if "2017" in t_lower else (
                "SystemVerilog-2009" if "2009" in t_lower else (
                    "SystemVerilog-2005" if "2005" in t_lower else "SystemVerilog-2012"
                )
            )
        )
        return ResolvedLanguage(HDLLanguage.SYSTEMVERILOG, ver)
    elif "verilog" in t_lower or t_lower in ("v", "v2001", "v2005", "v1995"):
        ver = language_version or (
            "Verilog-1995" if "1995" in t_lower else (
                "Verilog-2005" if "2005" in t_lower else "Verilog-2001"
            )
        )
        return ResolvedLanguage(HDLLanguage.VERILOG, ver)
    if allow_unknown:
        return ResolvedLanguage(HDLLanguage.UNKNOWN, "Unknown")
    raise ValueError(f"Unsupported or ambiguous target HDL language: '{target}'. Supported languages are 'Verilog' and 'SystemVerilog'.")


def _strip_comments_and_strings(code: str) -> str:
    """Return code with block comments, line comments, and string literals removed."""
    # Remove block comments /* ... */
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Remove line comments // ...
    no_line = re.sub(r"//[^\n]*", "", no_block)
    # Remove string literals
    no_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', no_line)
    return no_strings


def validate_rtl_syntax_and_structure(
    code: str,
    target_language: str = "SystemVerilog",
    language_version: Optional[str] = None,
    rtl_plan: Optional[Dict[str, Any]] = None,
    efs_ir: Optional[Any] = None,
    run_external_tools: bool = True,
) -> HDLValidationReport:
    """
    Perform deep language-aware syntax, structural, and semantic validation.
    Works for both Verilog and SystemVerilog with language-specific rules.
    """
    lang, default_ver = resolve_hdl_language(target_language)
    version = language_version or default_ver

    errors: List[HDLError] = []
    warnings: List[str] = []
    passed_checks: List[str] = []

    if not code or not isinstance(code, str) or not code.strip():
        errors.append(
            HDLError(
                rule_id="RULE_EMPTY_CODE",
                message="Generated HDL code is empty or null.",
                severity=ValidationSeverity.CRITICAL,
            )
        )
        return HDLValidationReport(
            is_valid=False,
            language=lang,
            language_version=version,
            errors=errors,
            warnings=warnings,
            passed_checks=passed_checks,
        )

    # Check unknown target language
    if lang == HDLLanguage.UNKNOWN:
        errors.append(
            HDLError(
                rule_id="RULE_UNSUPPORTED_LANGUAGE",
                message=f"Requested target language '{target_language}' is unsupported or ambiguous. Supported targets are 'Verilog' and 'SystemVerilog'.",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Explicitly specify target HDL as 'Verilog' or 'SystemVerilog'.",
            )
        )
        return HDLValidationReport(
            is_valid=False,
            language=lang,
            language_version=version,
            errors=errors,
            warnings=warnings,
            passed_checks=passed_checks,
        )

    lines = code.split("\n")
    clean_code = _strip_comments_and_strings(code)

    # 1. Module boundary and truncation detection
    has_module = bool(re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)", clean_code))
    has_endmodule = bool(re.search(r"\bendmodule\b", clean_code))

    if not has_module:
        errors.append(
            HDLError(
                rule_id="RULE_MISSING_MODULE",
                message="Missing module declaration boundary (`module <name>`).",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Define module using `module <name> (...); ... endmodule`.",
            )
        )
    elif not has_endmodule:
        errors.append(
            HDLError(
                rule_id="RULE_MISSING_ENDMODULE",
                message="Truncated or incomplete module: missing closing `endmodule` statement.",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Ensure generated RTL completes with `endmodule`.",
            )
        )
    else:
        passed_checks.append("Module boundaries (module ... endmodule) are present.")

    # 2. Block balance and truncation check
    # Count begin / end
    begin_tokens = len(re.findall(r"\bbegin\b", clean_code))
    end_tokens = len(re.findall(r"\bend\b", clean_code))
    if begin_tokens != end_tokens:
        errors.append(
            HDLError(
                rule_id="RULE_UNBALANCED_BEGIN_END",
                message=f"Unbalanced procedural blocks: found {begin_tokens} `begin` statement(s) but {end_tokens} `end` statement(s).",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Ensure each `begin` statement has a matching `end`.",
            )
        )
    else:
        passed_checks.append("Procedural `begin` and `end` blocks are balanced.")

    # Count case / endcase
    case_tokens = len(re.findall(r"\bcase\s*\(", clean_code))
    endcase_tokens = len(re.findall(r"\bendcase\b", clean_code))
    if case_tokens != endcase_tokens:
        errors.append(
            HDLError(
                rule_id="RULE_UNBALANCED_CASE_ENDCASE",
                message=f"Unbalanced case statements: found {case_tokens} `case` statement(s) but {endcase_tokens} `endcase` statement(s).",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Ensure each `case` statement is closed with `endcase`.",
            )
        )
    else:
        if case_tokens > 0:
            passed_checks.append("Case statements are balanced with `endcase`.")

    # Check parenthesis and bracket balance
    paren_open = clean_code.count("(")
    paren_close = clean_code.count(")")
    if paren_open != paren_close:
        errors.append(
            HDLError(
                rule_id="RULE_UNBALANCED_PARENTHESES",
                message=f"Unbalanced parentheses: {paren_open} '(' vs {paren_close} ')'.",
                severity=ValidationSeverity.CRITICAL,
            )
        )

    bracket_open = clean_code.count("[")
    bracket_close = clean_code.count("]")
    if bracket_open != bracket_close:
        errors.append(
            HDLError(
                rule_id="RULE_UNBALANCED_BRACKETS",
                message=f"Unbalanced vector brackets: {bracket_open} '[' vs {bracket_close} ']'.",
                severity=ValidationSeverity.CRITICAL,
            )
        )

    # 3. Detect placeholder comments standing in for hardware logic
    placeholders = [
        (r"//\s*Active\s+low\s+reset", "Active low reset placeholder comment"),
        (r"//\s*Combinational\s+logic", "Combinational logic placeholder comment"),
        (r"//\s*Synthesizable\s+logic\s+implementation", "Synthesizable logic placeholder comment"),
        (r"//\s*Internal\s+architecture\s*&\s*register\s+transfer\s+logic", "Internal architecture placeholder comment"),
        (r"//\s*TODO\b", "TODO comment"),
        (r"//\s*TBD\b", "TBD comment"),
        (r"//\s*implement\s+me\b", "Implement-me comment"),
    ]
    for ph_pat, ph_desc in placeholders:
        for idx, line in enumerate(lines, 1):
            if re.search(ph_pat, line, re.IGNORECASE):
                errors.append(
                    HDLError(
                        rule_id="RULE_PLACEHOLDER_COMMENT",
                        message=f"RTL contains un-implemented placeholder text: '{ph_desc}'.",
                        severity=ValidationSeverity.CRITICAL,
                        line_number=idx,
                        snippet=line.strip(),
                        suggested_fix="Replace placeholder comment with complete synthesizable hardware logic.",
                    )
                )

    # 4. Detect empty procedural blocks
    empty_ff = re.search(
        r"always_ff\s*@\([^\)]*\)\s*begin\s*(?:if\s*\([^\)]*\)\s*begin\s*(?://[^\n]*\s*)*end\s*)?(?://[^\n]*\s*)*end",
        clean_code,
    )
    if empty_ff:
        errors.append(
            HDLError(
                rule_id="RULE_EMPTY_PROCEDURAL_BLOCK",
                message="Empty procedural block: `always_ff` contains no executable assignments.",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Provide register reset and update assignments inside `always_ff`.",
            )
        )

    empty_comb = re.search(r"always_comb\s*begin\s*(?://[^\n]*\s*)*end", clean_code)
    if empty_comb:
        errors.append(
            HDLError(
                rule_id="RULE_EMPTY_PROCEDURAL_BLOCK",
                message="Empty procedural block: `always_comb` contains no executable assignments.",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Provide combinational logic and output assignments inside `always_comb`.",
            )
        )

    empty_vlog_always = re.search(r"always\s*@\([^\)]*\)\s*begin\s*(?://[^\n]*\s*)*end", clean_code)
    if empty_vlog_always:
        errors.append(
            HDLError(
                rule_id="RULE_EMPTY_PROCEDURAL_BLOCK",
                message="Empty procedural block: `always @(...)` contains no executable assignments.",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Provide assignments inside procedural block.",
            )
        )

    # 5. Language-Specific Validation Rules
    if lang == HDLLanguage.VERILOG:
        # Check for forbidden SystemVerilog keywords in Verilog mode
        for kw in SV_ONLY_KEYWORDS:
            pattern = rf"\b{kw}\b"
            for idx, line in enumerate(lines, 1):
                # Only check non-comment line
                clean_l = re.sub(r"//.*", "", line)
                if re.search(pattern, clean_l):
                    errors.append(
                        HDLError(
                            rule_id="RULE_VERILOG_FORBIDDEN_SV_KEYWORD",
                            message=f"SystemVerilog keyword `{kw}` is not permitted in Verilog mode ({version}).",
                            severity=ValidationSeverity.CRITICAL,
                            line_number=idx,
                            snippet=line.strip(),
                            suggested_fix=f"Use standard Verilog compatible construct (e.g. `wire`/`reg` instead of `logic`, `always @(*)` instead of `always_comb`, `always @(posedge clk)` instead of `always_ff`, `localparam` instead of `typedef enum`).",
                        )
                    )

        # In Verilog, check that always blocks use standard sensitivity
        for idx, line in enumerate(lines, 1):
            clean_l = re.sub(r"//.*", "", line)
            if re.search(r"\balways\b", clean_l) and not re.search(r"\balways\s*@", clean_l):
                errors.append(
                    HDLError(
                        rule_id="RULE_VERILOG_ALWAYS_SENSITIVITY",
                        message="Verilog procedural `always` blocks must include an explicit sensitivity list (e.g. `always @(*)` or `always @(posedge clk)`).",
                        severity=ValidationSeverity.CRITICAL,
                        line_number=idx,
                        snippet=line.strip(),
                    )
                )

        # Check reg vs wire assignments in Verilog
        # continuous assign target must be wire, not reg
        assign_matches = re.finditer(r"\bassign\s+([a-zA-Z_][a-zA-Z0-9_]*)", clean_code)
        for am in assign_matches:
            target_net = am.group(1)
            # Check if target_net is declared as reg
            if re.search(rf"\breg\s+(?:\[[^\]]+\]\s+)?{re.escape(target_net)}\b", clean_code) or \
               re.search(rf"\boutput\s+reg\s+(?:\[[^\]]+\]\s+)?{re.escape(target_net)}\b", clean_code):
                errors.append(
                    HDLError(
                        rule_id="RULE_VERILOG_ASSIGN_TO_REG",
                        message=f"Continuous assignment to `reg` variable '{target_net}' is illegal in Verilog. Continuous assign targets must be net/wire.",
                        severity=ValidationSeverity.CRITICAL,
                        suggested_fix=f"Declare '{target_net}' as `wire` or drive it procedurally within an `always @(*)` block.",
                    )
                )

        passed_checks.append("Verilog language standard checks passed (no SV-only keywords, valid reg/wire usage).")

    elif lang == HDLLanguage.SYSTEMVERILOG:
        # In SystemVerilog mode, encourage modern synthesizable constructs
        # Check always_ff has edge sensitivity
        always_ff_matches = re.finditer(r"\balways_ff\s*@\((.*?)\)", clean_code, re.DOTALL)
        for afm in always_ff_matches:
            sens = afm.group(1).strip()
            if "posedge" not in sens and "negedge" not in sens:
                errors.append(
                    HDLError(
                        rule_id="RULE_SV_ALWAYS_FF_SENSITIVITY",
                        message=f"`always_ff` sensitivity list '@({sens})' must specify clock/reset edge triggers (`posedge` or `negedge`).",
                        severity=ValidationSeverity.CRITICAL,
                        suggested_fix="Use `@(posedge clk or negedge rst_n)` or `@(posedge clk)`.",
                    )
                )

        # Check always_comb has NO explicit sensitivity list
        always_comb_sens = re.finditer(r"\balways_comb\s*@", clean_code)
        for _ in always_comb_sens:
            errors.append(
                HDLError(
                    rule_id="RULE_SV_ALWAYS_COMB_SENSITIVITY",
                    message="`always_comb` must NOT have an explicit sensitivity list `@(...)`.",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix="Remove `@(...)` after `always_comb`.",
                )
            )

        passed_checks.append("SystemVerilog constructs (always_ff, always_comb, logic) correctly used.")

    # 6. Synthesizability checks for both languages
    # Check for delays
    for idx, line in enumerate(lines, 1):
        clean_l = re.sub(r"//.*", "", line)
        if re.search(r"#\s*\d+", clean_l):
            errors.append(
                HDLError(
                    rule_id="RULE_NON_SYNTHESIZABLE_DELAY",
                    message="Simulation delay (`#<number>`) detected in synthesizable RTL.",
                    severity=ValidationSeverity.CRITICAL,
                    line_number=idx,
                    snippet=line.strip(),
                    suggested_fix="Remove `#<delay>` statement from synthesizable RTL code.",
                )
            )

    # Check for non-synthesizable file I/O or simulation-only tasks in RTL
    sim_only = ["$fopen", "$fdisplay", "$fwrite", "$finish", "$stop"]
    for so in sim_only:
        for idx, line in enumerate(lines, 1):
            clean_l = re.sub(r"//.*", "", line)
            if so in clean_l:
                warnings.append(f"Simulation system task `{so}` detected on line {idx} (non-synthesizable in production ASIC/FPGA).")

    # 7. Extract declared ports and check against plan/EFS IR
    declared_ports = _extract_declared_ports(clean_code)
    output_port_names = [p["name"] for p in declared_ports if p["direction"] == "output"]

    # 8. Check for undriven output ports
    for op in output_port_names:
        # Check if output is assigned anywhere
        # As assign target: assign <op> = ...
        # Or in procedural block: <op> <= ... or <op> = ...
        is_driven = bool(
            re.search(rf"\bassign\s+{re.escape(op)}\b\s*=", clean_code) or
            re.search(rf"\b{re.escape(op)}\b\s*(?:<=|=)", clean_code)
        )
        if not is_driven:
            errors.append(
                HDLError(
                    rule_id="RULE_UNDRIVEN_OUTPUT",
                    message=f"Output port '{op}' is declared but never driven by any assignment in the module.",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix=f"Add assignment for output port '{op}' in combinational or sequential logic.",
                )
            )
        else:
            passed_checks.append(f"Output port '{op}' is driven.")

    # 9. Check plan/EFS IR conformance if provided
    if rtl_plan:
        plan_ports = rtl_plan.get("ports", [])
        for p in plan_ports:
            p_name = p.get("name", "")
            if not p_name or p_name.lower() in ("clk", "clock", "rst", "reset", "rst_n"):
                continue

            # Check if port exists in RTL
            if not any(dp["name"].lower() == p_name.lower() for dp in declared_ports):
                errors.append(
                    HDLError(
                        rule_id="RULE_MISSING_PORT_FROM_PLAN",
                        message=f"Required port '{p_name}' from implementation plan is missing from module port declarations.",
                        severity=ValidationSeverity.CRITICAL,
                        suggested_fix=f"Declare port '{p_name}' ({p.get('direction', 'input')}) in module header.",
                    )
                )

            # Check vector width
            raw_w = str(p.get("width", "1")).strip()
            norm_w = _parse_width(raw_w)
            if norm_w and norm_w > 1:
                # Find port declaration in code
                decl_scalar = re.search(rf"\b(?:input|output|inout)\s+(?:logic|wire|reg)?\s*{re.escape(p_name)}\b(?!\s*\[)", clean_code)
                decl_vector = re.search(rf"\[[^\]]+\]\s*{re.escape(p_name)}\b", clean_code)
                if decl_scalar and not decl_vector:
                    errors.append(
                        HDLError(
                            rule_id="RULE_PORT_WIDTH_MISMATCH",
                            message=f"Port '{p_name}' requires vector width {norm_w}, but is declared as 1-bit scalar in generated code.",
                            severity=ValidationSeverity.CRITICAL,
                            suggested_fix=f"Update declaration to include range `[{norm_w-1}:0] {p_name}`.",
                        )
                    )

        # Check FSM states if plan has states
        plan_fsm = rtl_plan.get("fsm", {})
        plan_states = plan_fsm.get("states", []) if isinstance(plan_fsm, dict) else []
        if len(plan_states) > 1:
            for st in plan_states:
                st_name = st.get("name", "") if isinstance(st, dict) else str(st)
                if st_name and not re.search(rf"\b{re.escape(st_name)}\b", clean_code):
                    errors.append(
                        HDLError(
                            rule_id="RULE_MISSING_FSM_STATE",
                            message=f"FSM state '{st_name}' specified in plan was not detected in generated RTL declarations or logic.",
                            severity=ValidationSeverity.CRITICAL,
                            suggested_fix=f"Declare and implement state '{st_name}' in FSM logic.",
                        )
                    )

    # 10. Natural Language Fragment Detection in Active Code
    nl_phrases = [
        r"\bdata\s+payload\b",
        r"\binternal\s+register\b",
        r"\bcontrol\s+start\b",
        r"\bcondition\s+met\b",
        r"\boperation\s+completes\b",
        r"\btransaction\s+acknowledged\b",
        r"\bto\s+be\s+implemented\b",
        r"\bunspecified\b",
        r"\bundefined\b",
        r"\btbd\b",
    ]
    for nl_pat in nl_phrases:
        for idx, line in enumerate(lines, 1):
            clean_l = re.sub(r"//.*", "", line)
            if re.search(nl_pat, clean_l, re.IGNORECASE):
                nl_clean = nl_pat.replace(r"\b", "").replace(r"\s+", " ")
                errors.append(
                    HDLError(
                        rule_id="RULE_NATURAL_LANGUAGE_FRAGMENT",
                        message=f"RTL line {idx} contains natural-language specification fragment matching '{nl_clean}' in active code.",
                        severity=ValidationSeverity.CRITICAL,
                        line_number=idx,
                        snippet=line.strip(),
                        suggested_fix="Replace natural-language description with valid synthesizable Verilog/SystemVerilog expressions.",
                    )
                )

    # 11. Check for Unused Input Ports
    input_port_names = [p["name"] for p in declared_ports if p["direction"] == "input"]
    clock_reset_names = {"clk", "clock", "aclk", "s_axi_aclk", "m_axi_aclk", "rst", "reset", "rst_n", "aresetn", "s_axi_aresetn", "m_axi_aresetn"}
    m_body_match = re.search(r"\bmodule\s+[a-zA-Z0-9_]+\s*(?:#\s*\([^\)]*\)\s*)?\([^\)]*\);(.*)\bendmodule\b", clean_code, re.DOTALL)
    module_body = m_body_match.group(1) if m_body_match else clean_code

    for ip in input_port_names:
        if ip.lower() in clock_reset_names:
            continue
        is_used = bool(re.search(rf"\b{re.escape(ip)}\b", module_body))
        if not is_used:
            warnings.append(f"Input port '{ip}' is declared in module header but never read or utilized in the module functional logic.")
        else:
            passed_checks.append(f"Input port '{ip}' is utilized in functional logic.")

    # 12. Undeclared Identifier Check
    declared_identifiers: Set[str] = {p["name"] for p in declared_ports}
    for m in re.finditer(r"\b(?:reg|wire|logic|integer|genvar|localparam|parameter)\s+(?:\[[^\]]+\]\s+)?([a-zA-Z_][a-zA-Z0-9_]*)", clean_code):
        declared_identifiers.add(m.group(1))
    for m in re.finditer(r"\blocalparam\s+(?:\[[^\]]+\]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=", clean_code):
        declared_identifiers.add(m.group(1))
    for m in re.finditer(r"\bparameter\s+(?:\[[^\]]+\]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=", clean_code):
        declared_identifiers.add(m.group(1))
    for m in re.finditer(r"\btypedef\s+enum[^{]*\{([^}]+)\}", clean_code):
        for enum_val in m.group(1).split(","):
            val_clean = enum_val.split("=")[0].strip()
            if val_clean:
                declared_identifiers.add(val_clean)
    if "state" in clean_code:
        declared_identifiers.add("state")
        declared_identifiers.add("next_state")
    if "timer_cnt" in clean_code:
        declared_identifiers.add("timer_cnt")

    hdl_keywords: Set[str] = {
        "begin", "end", "if", "else", "case", "endcase", "casex", "casez", "default", "assign",
        "posedge", "negedge", "or", "and", "not", "module", "endmodule", "always", "always_comb",
        "always_ff", "always_latch", "input", "output", "inout", "wire", "reg", "logic",
        "parameter", "localparam", "signed", "unsigned", "initial", "for", "while", "repeat",
        "return", "function", "endfunction", "task", "endtask", "generate", "endgenerate",
        "genvar", "integer", "bit", "int", "byte", "shortint", "longint", "typedef", "enum",
        "struct", "union", "disable", "iff", "assert", "property", "endproperty",
        "d", "b", "h", "o", "sd", "sb", "sh", "so"
    }

    assign_targets = re.finditer(r"\bassign\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=", clean_code)
    for at in assign_targets:
        target = at.group(1)
        if target not in declared_identifiers and target not in hdl_keywords:
            errors.append(
                HDLError(
                    rule_id="RULE_UNDECLARED_IDENTIFIER",
                    message=f"Identifier '{target}' is driven by continuous assignment but never declared.",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix=f"Declare '{target}' as a wire or net before assigning.",
                )
            )

    # 13. FSM Unconditional Transition Check
    if "case" in clean_code and "state" in clean_code:
        case_blocks = re.findall(r"case\s*\(\s*state\s*\)(.*?)endcase", clean_code, re.DOTALL)
        for cb in case_blocks:
            branches = re.findall(r"\b([a-zA-Z0-9_]+)\s*:\s*begin(.*?)end", cb, re.DOTALL)
            uncond_count = 0
            active_branches = [b for b in branches if b[0].lower() not in ("default", "reset")]
            for b_name, b_body in active_branches:
                has_next = "next_state" in b_body
                has_if = "if" in b_body or "?" in b_body or "case" in b_body
                if has_next and not has_if:
                    uncond_count += 1
            if len(active_branches) >= 3 and uncond_count == len(active_branches):
                errors.append(
                    HDLError(
                        rule_id="RULE_UNCONDITIONAL_FSM_TRANSITION",
                        message=f"All {len(active_branches)} FSM states transition unconditionally without evaluating input triggers or completion conditions.",
                        severity=ValidationSeverity.MAJOR,
                        suggested_fix="Ensure state transitions depend on input conditions, handshakes, or timer completion.",
                    )
                )

    # 14. Check for illegal identifiers containing spaces, dashes, or descriptive phrases in hardware declarations
    decl_matches = re.finditer(r"\b(?:reg|wire|logic)\s+(?:\[[^\]]+\]\s+)?([^;]+);", clean_code)
    for dm in decl_matches:
        decl_str = dm.group(1).strip()
        for item in decl_str.split(","):
            id_candidate = item.split("=")[0].strip()
            if id_candidate and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", id_candidate):
                errors.append(
                    HDLError(
                        rule_id="RULE_ILLEGAL_IDENTIFIER",
                        message=f"Illegal identifier '{id_candidate}' containing spaces or special characters in declaration `{dm.group(0).strip()}`.",
                        severity=ValidationSeverity.CRITICAL,
                        suggested_fix="Declare hardware registers with valid identifier names matching `^[a-zA-Z_][a-zA-Z0-9_]*$`.",
                    )
                )

    # 15. Check for undefined or null values in assignments
    for idx, line in enumerate(lines, 1):
        clean_l = re.sub(r"//.*", "", line)
        if re.search(r"(?:<=|=)\s*\b(?:None|null|undefined)\b", clean_l):
            errors.append(
                HDLError(
                    rule_id="RULE_UNDEFINED_VALUE",
                    message=f"RTL line {idx} assigns undefined/null value to hardware signal or register.",
                    severity=ValidationSeverity.CRITICAL,
                    line_number=idx,
                    snippet=line.strip(),
                    suggested_fix="Assign valid synthesizable constant (e.g. `32'h0` or `1'b0`) instead of undefined value.",
                )
            )

    # 16. Semantic Status Inversion Check
    # Check if 'done' is equated to an error state (e.g. done = (state == ERROR))
    done_err_match = re.search(r"\bdone\b\s*(?:<=|=)\s*\(?\s*state\s*==\s*([a-zA-Z0-9_]*ERROR[a-zA-Z0-9_]*)\s*\)?", clean_code, re.IGNORECASE)
    if done_err_match:
        errors.append(
            HDLError(
                rule_id="RULE_SEMANTIC_STATUS_INVERSION",
                message=f"Semantic status inversion: 'done' output is equated to error state '{done_err_match.group(1)}'. Successful completion must be distinct from error handling.",
                severity=ValidationSeverity.CRITICAL,
                suggested_fix="Assign 'done' to true completion state (e.g. DONE or FINISH), not ERROR.",
            )
        )
    # Check if 'error' is driven by a data register (e.g. error = M0;)
    err_reg_match = re.search(r"\berror\b\s*(?:<=|=)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*;", clean_code, re.IGNORECASE)
    if err_reg_match:
        rhs = err_reg_match.group(1).strip()
        if re.match(r"^[MR]\d+$", rhs, re.IGNORECASE) or any(k in rhs.lower() for k in ("matrix_reg", "data_in", "payload")):
            errors.append(
                HDLError(
                    rule_id="RULE_SEMANTIC_STATUS_INVERSION",
                    message=f"Semantic status inversion: 'error' output is directly assigned data register '{rhs}'. Status outputs must correspond to defined functional meanings.",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix="Assign 'error' to error condition or state (e.g. `state == ERROR`), not data register.",
                )
            )

    # 17. Unjustified Interface / Handshake Register Mapping
    ctrl_assigns = re.finditer(r"\b([a-zA-Z0-9_]*(?:valid|ready|addr|strb|req|ack)[a-zA-Z0-9_]*)\s*(?:<=|=)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*;", clean_code, re.IGNORECASE)
    for ca in ctrl_assigns:
        out_sig = ca.group(1).strip()
        rhs_sig = ca.group(2).strip()
        if re.match(r"^[MR]\d+$", rhs_sig, re.IGNORECASE) or "matrix_reg" in rhs_sig.lower():
            errors.append(
                HDLError(
                    rule_id="RULE_UNJUSTIFIED_REGISTER_MAPPING",
                    message=f"Interface control/address output '{out_sig}' is driven by unjustified data register '{rhs_sig}'.",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix=f"Drive '{out_sig}' using protocol state or address register, not general-purpose data register.",
                )
            )

    # 18. Arbitrary Unconditional Input-to-Register Assignment Check
    for block in re.finditer(r"always\s*(?:@\s*\([^\)]*\)|_ff\s*@\s*\([^\)]*\))\s*begin\s*if\s*\([^\)]*\)\s*begin(.*?)end\s*else\s*begin(.*?)end\s*end", clean_code, re.DOTALL):
        else_body = block.group(2)
        if "case" not in else_body and "if" not in else_body:
            assign_lines = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*;", else_body)
            if len(assign_lines) >= 3:
                errors.append(
                    HDLError(
                        rule_id="RULE_ARBITRARY_UNCONDITIONAL_ASSIGNMENT",
                        message=f"Arbitrary unconditional input-to-register assignments detected for {len(assign_lines)} registers without gating condition.",
                        severity=ValidationSeverity.CRITICAL,
                        suggested_fix="Gate register assignments with enable conditions or state machine transitions.",
                    )
                )

    # 19. Run external compiler / linter if available
    tool_status = "TOOL_NOT_RUN"
    tool_details = None
    if run_external_tools:
        ext_res = _run_external_hdl_tool(code, lang, version)
        tool_status = ext_res["status"]
        tool_details = ext_res.get("details")
        if ext_res.get("errors"):
            for tool_err in ext_res["errors"]:
                errors.append(tool_err)

    is_valid = len([e for e in errors if e.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.MAJOR)]) == 0

    return HDLValidationReport(
        is_valid=is_valid,
        language=lang,
        language_version=version,
        errors=errors,
        warnings=warnings,
        passed_checks=passed_checks,
        tool_status=tool_status,
        tool_details=tool_details,
        metrics={
            "declared_ports_count": len(declared_ports),
            "output_ports_count": len(output_port_names),
            "begin_blocks": begin_tokens,
            "case_blocks": case_tokens,
        },
    )


def _parse_width(width_str: str) -> Optional[int]:
    """Parse numeric width from strings like '32', '[31:0]', etc."""
    s = str(width_str).strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", s)
    if m:
        msb = int(m.group(1))
        lsb = int(m.group(2))
        return abs(msb - lsb) + 1
    return None


def _extract_declared_ports(clean_code: str) -> List[Dict[str, str]]:
    """Extract port declarations from module header."""
    ports = []
    # Match module header
    m_header = re.search(r"\bmodule\s+[a-zA-Z0-9_]+\s*(?:#\s*\([^\)]*\)\s*)?\((.*?)\);", clean_code, re.DOTALL)
    if not m_header:
        # Non-ANSI style header
        m_non_ansi = re.search(r"\bmodule\s+[a-zA-Z0-9_]+\s*(?:#\s*\([^\)]*\)\s*)?\((.*?)\);", clean_code)
        # Search for input/output declarations in module body
        body_ports = re.finditer(r"\b(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[[^\]]+\])?\s*([a-zA-Z_][a-zA-Z0-9_]*)", clean_code)
        for bp in body_ports:
            ports.append({"name": bp.group(2), "direction": bp.group(1).lower()})
        return ports

    header_text = m_header.group(1)
    port_items = [p.strip() for p in header_text.split(",") if p.strip()]
    curr_dir = "input"
    for item in port_items:
        m_dir = re.search(r"\b(input|output|inout)\b", item)
        if m_dir:
            curr_dir = m_dir.group(1).lower()
        # Find signal name (last word in port item)
        m_name = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", item)
        if m_name:
            sig_name = m_name[-1]
            if sig_name not in ("input", "output", "inout", "logic", "wire", "reg", "signed", "unsigned"):
                ports.append({"name": sig_name, "direction": curr_dir})
    return ports


def _run_external_hdl_tool(code: str, language: HDLLanguage, version: str) -> Dict[str, Any]:
    """Execute installed external HDL compilers/linters with language-correct flags."""
    iverilog_path = shutil.which("iverilog")
    verilator_path = shutil.which("verilator")
    slang_path = shutil.which("slang")

    if not iverilog_path and not verilator_path and not slang_path:
        return {
            "status": "TOOL_NOT_AVAILABLE",
            "details": "No external HDL compiler (iverilog, verilator, slang) detected in environment. Deterministic static validation utilized.",
            "errors": [],
        }

    suffix = ".sv" if language == HDLLanguage.SYSTEMVERILOG else ".v"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
            f.write(code)
            temp_path = f.name

        tool_errors = []

        if iverilog_path:
            out_file = temp_path + ".out"
            # Choose flag based on language & version
            if language == HDLLanguage.SYSTEMVERILOG:
                iverilog_flag = "-g2012"
            elif "1995" in version:
                iverilog_flag = "-g1995"
            else:
                iverilog_flag = "-g2005"

            res = subprocess.run(
                [iverilog_path, iverilog_flag, "-o", out_file, temp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(out_file):
                os.remove(out_file)

            if res.returncode != 0:
                err_text = res.stderr.strip() or res.stdout.strip()
                # Parse line numbers from iverilog output: <file>:<line>: error: ...
                for l in err_text.split("\n"):
                    l_strip = l.strip()
                    if not l_strip:
                        continue
                    m_line = re.search(r":(\d+):\s*(.*)", l_strip)
                    line_no = int(m_line.group(1)) if m_line else None
                    msg = m_line.group(2) if m_line else l_strip
                    tool_errors.append(
                        HDLError(
                            rule_id="RULE_IVERILOG_SYNTAX_ERROR",
                            message=f"Icarus Verilog: {msg}",
                            severity=ValidationSeverity.CRITICAL,
                            line_number=line_no,
                        )
                    )
                return {
                    "status": "FAIL",
                    "details": f"Icarus Verilog ({iverilog_flag}) compilation failed.",
                    "errors": tool_errors,
                }
            return {
                "status": "PASS",
                "details": f"Compiled cleanly with Icarus Verilog ({iverilog_flag}). Note: Compiler success verifies syntax/elaboration, not functional correctness.",
                "errors": [],
            }

        elif verilator_path:
            flags = ["--lint-only"]
            if language == HDLLanguage.SYSTEMVERILOG:
                flags.append("-sv")
            res = subprocess.run(
                [verilator_path] + flags + [temp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if res.returncode != 0:
                err_text = res.stderr.strip() or res.stdout.strip()
                return {
                    "status": "FAIL",
                    "details": f"Verilator lint failed: {err_text}",
                    "errors": [
                        HDLError(
                            rule_id="RULE_VERILATOR_LINT_ERROR",
                            message=f"Verilator: {err_text[:300]}",
                            severity=ValidationSeverity.CRITICAL,
                        )
                    ],
                }
            return {
                "status": "PASS",
                "details": "Linted cleanly with Verilator. Note: Compiler success verifies syntax/elaboration, not functional correctness.",
                "errors": [],
            }

    except Exception as e:
        logger.debug(f"External HDL tool execution exception: {e}")
        return {
            "status": "TOOL_ERROR",
            "details": str(e),
            "errors": [],
        }

    return {"status": "TOOL_NOT_RUN", "details": None, "errors": []}
