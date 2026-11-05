"""Verilog Expression and Action Parser with Precedence Parsing and Symbol Resolution.

Supports:
- Sized literals (e.g. 1'b1, 8'hFF, 32'd10, 4'b00_10)
- Unsized literals (e.g. 0, 10, '0, '1, 'hFF)
- Identifiers with declaration / symbol table verification
- Unary operators: !, ~, +, -, and reductions (&, ~&, |, ~|, ^, ~^, ^~)
- Binary operators with standard Verilog precedence:
  *, /, %
  +, -
  <<, >>, <<<, >>>
  <, <=, >, >=
  ==, !=, ===, !==, ==?, !=?
  &
  ^, ~^, ^~
  |
  &&
  ||
- Conditional / Ternary: ?:
- Bit-select [i] and part-select [msb:lsb], [base +: width], [base -: width]
- Concatenation {a, b, ...} and replication {N{a, b}}
- Action statements: target <= expr; or target = expr;
- Rejection of Verilog reserved keywords and unknown identifiers
- Detection of comparison vs assignment confusion
"""

import re
from typing import List, Dict, Set, Optional, Tuple, Any, Union


VERILOG_KEYWORDS = {
    "always", "always_comb", "always_ff", "always_latch", "and", "assert",
    "assign", "automatic", "begin", "bit", "buf", "bufif0", "bufif1",
    "byte", "case", "casex", "casez", "cell", "chandle", "class", "clocking",
    "config", "const", "constraint", "context", "continue", "cover",
    "covergroup", "coverpoint", "cross", "deassign", "default", "defparam",
    "design", "disable", "dist", "do", "edge", "else", "end", "endcase",
    "endclass", "endclocking", "endconfig", "endfunction", "endgenerate",
    "endgroup", "endinterface", "endmodule", "endpackage", "endprimitive",
    "endprogram", "endproperty", "endsequence", "endspecify", "endtable",
    "endtask", "enum", "event", "expect", "export", "extends", "extern",
    "final", "first_match", "for", "force", "foreach", "forever", "fork",
    "forkjoin", "function", "generate", "genvar", "highz0", "highz1", "if",
    "iff", "ifnone", "import", "incdir", "include", "initial", "inout",
    "input", "inside", "instance", "int", "integer", "interface", "intersect",
    "join", "join_any", "join_none", "large", "let", "liblist", "library",
    "local", "localparam", "logic", "longint", "matches", "medium", "modport",
    "module", "nand", "negedge", "new", "nmos", "nor", "noshowcancelled",
    "not", "notif0", "notif1", "null", "or", "output", "package", "packed",
    "parameter", "pmos", "posedge", "primitive", "priority", "program",
    "property", "protected", "pull0", "pull1", "pulldown", "pullup",
    "pulsestyle_onevent", "pulsestyle_ondetect", "pure", "rand", "randc",
    "randcase", "randsequence", "rcmos", "real", "realtime", "ref", "reg",
    "release", "repeat", "return", "rnmos", "rpmos", "rtran", "rtranif0",
    "rtranif1", "scalared", "sequence", "shortint", "shortreal", "showcancelled",
    "signed", "small", "soft", "solve", "specify", "specparam", "static",
    "string", "strong0", "strong1", "struct", "super", "supply0", "supply1",
    "table", "tagged", "task", "this", "throughout", "time", "timeprecision",
    "timeunit", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1", "triand",
    "trior", "trireg", "type", "typedef", "union", "unique", "unique0",
    "unsigned", "use", "uwire", "var", "vectored", "virtual", "void", "wait",
    "wait_order", "wand", "weak0", "weak1", "while", "wildcard", "wire",
    "with", "within", "wor", "xnor", "xor"
}

# English words commonly present in natural language prose conditions
PROSE_INDICATOR_WORDS = {
    "asserted", "deasserted", "assertion", "high", "low", "when", "whenever",
    "is", "are", "becomes", "active", "inactive", "valid", "invalid", "after",
    "cycles", "cycle", "clocks", "clock", "equals", "reaches", "completes",
    "completed", "finished", "timeout", "dwell", "single", "until", "unless",
    "received", "acknowledgement", "acknowledged", "condition", "undefined"
}


class TokenType:
    IDENTIFIER = "IDENTIFIER"
    SIZED_LITERAL = "SIZED_LITERAL"
    UNSIZED_INT = "UNSIZED_INT"
    UNSIZED_BASE = "UNSIZED_BASE"
    OPERATOR = "OPERATOR"
    DELIMITER = "DELIMITER"
    EOF = "EOF"


class Token:
    def __init__(self, type_: str, value: str, pos: int):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, pos={self.pos})"


class ExpressionTokenizer:
    """Tokenizer for Verilog expressions."""

    TOKEN_SPEC = [
        # Sized literal: e.g. 1'b1, 32'hDEAD_BEEF, 8'd255, 4'b10??
        ("SIZED_LITERAL", r"\b\d+\s*'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ\?]+"),
        # Unsized base literal: e.g. '0, '1, 'hFF, 'b1010
        ("UNSIZED_BASE", r"'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ\?]+|'[01xzXZ]"),
        # Multi-char operators
        ("MULTI_OP", r"===|!==|==\?|!=\?|<<<|>>>|<=|>=|==|!=|&&|\|\||<<|>>|~\^|\^~|~&|~\||\+:|-:"),
        # Unsized decimal integer
        ("UNSIZED_INT", r"\b\d+\b"),
        # Single-char operators
        ("SINGLE_OP", r"[!~+\-*/%&|^?:<>=]"),
        # Delimiters
        ("DELIMITER", r"[()\[\]{},;]"),
        # Identifiers
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_$]*"),
        # Whitespace
        ("SKIP", r"[ \t\r\n]+"),
        # Unknown character
        ("MISMATCH", r"."),
    ]

    MASTER_REGEX = re.compile("|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in TOKEN_SPEC))

    @classmethod
    def tokenize(cls, text: str) -> Tuple[List[Token], Optional[str]]:
        tokens: List[Token] = []
        for match in cls.MASTER_REGEX.finditer(text):
            kind = match.lastgroup
            val = match.group()
            pos = match.start()
            if kind == "SKIP":
                continue
            elif kind == "MISMATCH":
                return [], f"Unexpected character '{val}' at position {pos}"
            else:
                if kind in ("SIZED_LITERAL", "UNSIZED_BASE"):
                    val = re.sub(r"\s+", "", val)
                if kind in ("MULTI_OP", "SINGLE_OP"):
                    kind = TokenType.OPERATOR
                tokens.append(Token(kind, val, pos))
        tokens.append(Token(TokenType.EOF, "", len(text)))
        return tokens, None


class ASTNode:
    pass


class IdentifierNode(ASTNode):
    def __init__(self, name: str, select: Optional[ASTNode] = None):
        self.name = name
        self.select = select  # bit or part select node

    def __repr__(self):
        return f"Id({self.name}{f'[{self.select}]' if self.select else ''})"


class LiteralNode(ASTNode):
    def __init__(self, val: str, kind: str):
        self.val = val
        self.kind = kind

    def __repr__(self):
        return f"Lit({self.val})"


class UnaryOpNode(ASTNode):
    def __init__(self, op: str, expr: ASTNode):
        self.op = op
        self.expr = expr

    def __repr__(self):
        return f"Unary({self.op} {self.expr})"


class BinaryOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Binary({self.left} {self.op} {self.right})"


class TernaryOpNode(ASTNode):
    def __init__(self, cond: ASTNode, true_expr: ASTNode, false_expr: ASTNode):
        self.cond = cond
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"Ternary({self.cond} ? {self.true_expr} : {self.false_expr})"


class ConcatNode(ASTNode):
    def __init__(self, elements: List[ASTNode], repeat: Optional[ASTNode] = None):
        self.elements = elements
        self.repeat = repeat

    def __repr__(self):
        rep = f"{self.repeat} " if self.repeat else ""
        return f"Concat({{{rep}{', '.join(str(e) for e in self.elements)}}})"


class PartSelectNode(ASTNode):
    def __init__(self, left: ASTNode, right: Optional[ASTNode], kind: str = ":"):
        self.left = left
        self.right = right
        self.kind = kind  # ":", "+:", "-:" or "bit"

    def __repr__(self):
        if self.right is None:
            return f"{self.left}"
        return f"{self.left}{self.kind}{self.right}"


class ActionAssignmentNode(ASTNode):
    def __init__(self, target: IdentifierNode, op: str, expr: ASTNode):
        self.target = target
        self.op = op  # "<=" (non-blocking) or "=" (blocking)
        self.expr = expr

    def __repr__(self):
        return f"Assign({self.target} {self.op} {self.expr})"


class VerilogPrecedenceParser:
    """Precedence parser for Verilog expressions."""

    # Operator precedence (lowest to highest)
    PRECEDENCE = {
        "?": 1,
        "||": 2,
        "&&": 3,
        "|": 4, "~|": 4,
        "^": 5, "~^": 5, "^~": 5,
        "&": 6, "~&": 6,
        "==": 7, "!=": 7, "===": 7, "!==": 7, "==?": 7, "!=?": 7,
        "<": 8, "<=": 8, ">": 8, ">=": 8,
        "<<": 9, ">>": 9, "<<<": 9, ">>>": 9,
        "+": 10, "-": 10,
        "*": 11, "/": 11, "%": 11,
    }

    UNARY_OPS = {"!", "~", "+", "-", "&", "~&", "|", "~|", "^", "~^", "^~"}

    def __init__(self, tokens: List[Token], text: str = ""):
        self.tokens = tokens
        self.pos = 0
        self.text = text

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]

    def advance(self) -> Token:
        tok = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def peek(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return self.tokens[-1]

    def match(self, kind: str, val: Optional[str] = None) -> bool:
        c = self.current()
        if c.type == kind and (val is None or c.value == val):
            self.advance()
            return True
        return False

    def expect(self, kind: str, val: Optional[str] = None) -> Tuple[Optional[Token], Optional[str]]:
        c = self.current()
        if c.type == kind and (val is None or c.value == val):
            return self.advance(), None
        expected = val if val else kind
        return None, f"Expected '{expected}' but found '{c.value or c.type}' at position {c.pos}"

    def parse_expression(self, min_prec: int = 1) -> Tuple[Optional[ASTNode], Optional[str]]:
        left, err = self.parse_unary()
        if err:
            return None, err

        while True:
            tok = self.current()
            if tok.type != TokenType.OPERATOR:
                break
            op = tok.value
            prec = self.PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break

            self.advance()
            if op == "?":
                # Ternary operator: cond ? true_expr : false_expr
                true_expr, err = self.parse_expression(1)
                if err:
                    return None, err
                _, err = self.expect(TokenType.OPERATOR, ":")
                if err:
                    return None, f"Ternary operator missing ':' after '?' at position {tok.pos}"
                false_expr, err = self.parse_expression(1)
                if err:
                    return None, err
                left = TernaryOpNode(left, true_expr, false_expr)
            else:
                next_prec = prec + 1
                right, err = self.parse_expression(next_prec)
                if err:
                    return None, err
                left = BinaryOpNode(op, left, right)

        return left, None

    def parse_unary(self) -> Tuple[Optional[ASTNode], Optional[str]]:
        tok = self.current()
        if tok.type == TokenType.OPERATOR and tok.value in self.UNARY_OPS:
            self.advance()
            operand, err = self.parse_unary()
            if err:
                return None, err
            return UnaryOpNode(tok.value, operand), None
        return self.parse_primary()

    def parse_primary(self) -> Tuple[Optional[ASTNode], Optional[str]]:
        tok = self.current()

        if tok.type == TokenType.SIZED_LITERAL:
            self.advance()
            return LiteralNode(tok.value, "sized"), None

        if tok.type == TokenType.UNSIZED_BASE:
            self.advance()
            return LiteralNode(tok.value, "base"), None

        if tok.type == TokenType.UNSIZED_INT:
            self.advance()
            return LiteralNode(tok.value, "int"), None

        if tok.type == TokenType.IDENTIFIER:
            id_name = tok.value
            self.advance()
            # Check for bit-select or part-select [ ... ]
            if self.match(TokenType.DELIMITER, "["):
                select_node, err = self.parse_select()
                if err:
                    return None, err
                _, err = self.expect(TokenType.DELIMITER, "]")
                if err:
                    return None, err
                return IdentifierNode(id_name, select=select_node), None
            return IdentifierNode(id_name), None

        if self.match(TokenType.DELIMITER, "("):
            expr, err = self.parse_expression(1)
            if err:
                return None, err
            _, err = self.expect(TokenType.DELIMITER, ")")
            if err:
                return None, err
            return expr, None

        if self.match(TokenType.DELIMITER, "{"):
            return self.parse_concatenation()

        return None, f"Unexpected token '{tok.value or tok.type}' at position {tok.pos}"

    def parse_select(self) -> Tuple[Optional[ASTNode], Optional[str]]:
        left_expr, err = self.parse_expression(1)
        if err:
            return None, err

        tok = self.current()
        if tok.type == TokenType.OPERATOR and tok.value in (":", "+:", "-:"):
            op = tok.value
            self.advance()
            right_expr, err = self.parse_expression(1)
            if err:
                return None, err
            return PartSelectNode(left_expr, right_expr, kind=op), None
        return PartSelectNode(left_expr, None, kind="bit"), None

    def parse_concatenation(self) -> Tuple[Optional[ASTNode], Optional[str]]:
        # Peek if replication: e.g. { 4 { 1'b0 } }
        first_expr, err = self.parse_expression(1)
        if err:
            return None, err

        if self.match(TokenType.DELIMITER, "{"):
            # It's a replication: { repeat_count { elem1, elem2, ... } }
            repeat_count = first_expr
            inner_elems = []
            while True:
                elem, err = self.parse_expression(1)
                if err:
                    return None, err
                inner_elems.append(elem)
                if not self.match(TokenType.DELIMITER, ","):
                    break
            _, err = self.expect(TokenType.DELIMITER, "}")
            if err:
                return None, err
            _, err = self.expect(TokenType.DELIMITER, "}")
            if err:
                return None, err
            return ConcatNode(inner_elems, repeat=repeat_count), None
        else:
            # Regular concatenation { elem1, elem2, ... }
            elements = [first_expr]
            while self.match(TokenType.DELIMITER, ","):
                elem, err = self.parse_expression(1)
                if err:
                    return None, err
                elements.append(elem)
            _, err = self.expect(TokenType.DELIMITER, "}")
            if err:
                return None, err
            return ConcatNode(elements), None


class VerilogExpressionParser:
    """High-level Expression and Action Parser with Symbol Resolution."""

    @classmethod
    def collect_identifiers(cls, node: ASTNode) -> List[Tuple[str, Optional[ASTNode]]]:
        """Recursively collect all identifier references from an AST."""
        ids: List[Tuple[str, Optional[ASTNode]]] = []
        if isinstance(node, IdentifierNode):
            ids.append((node.name, node.select))
            if node.select:
                ids.extend(cls.collect_identifiers(node.select))
        elif isinstance(node, UnaryOpNode):
            ids.extend(cls.collect_identifiers(node.expr))
        elif isinstance(node, BinaryOpNode):
            ids.extend(cls.collect_identifiers(node.left))
            ids.extend(cls.collect_identifiers(node.right))
        elif isinstance(node, TernaryOpNode):
            ids.extend(cls.collect_identifiers(node.cond))
            ids.extend(cls.collect_identifiers(node.true_expr))
            ids.extend(cls.collect_identifiers(node.false_expr))
        elif isinstance(node, ConcatNode):
            if node.repeat:
                ids.extend(cls.collect_identifiers(node.repeat))
            for el in node.elements:
                ids.extend(cls.collect_identifiers(el))
        elif isinstance(node, PartSelectNode):
            if node.left:
                ids.extend(cls.collect_identifiers(node.left))
            if node.right:
                ids.extend(cls.collect_identifiers(node.right))
        elif isinstance(node, ActionAssignmentNode):
            ids.extend(cls.collect_identifiers(node.target))
            ids.extend(cls.collect_identifiers(node.expr))
        return ids

    @classmethod
    def check_prose(cls, text: str) -> Optional[str]:
        """Detect obvious natural language prose in condition text."""
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
        prose_matches = [w for w in words if w in PROSE_INDICATOR_WORDS]
        if len(prose_matches) >= 2 or any(p in text.lower() for p in ("asserted", "deasserted", "after 1 cycle", "dwell", "is high", "is low")):
            return f"Condition contains natural-language prose: '{text}'"
        return None

    @classmethod
    def parse_condition(
        cls,
        cond_text: str,
        declared_symbols: Set[str]
    ) -> Tuple[bool, Optional[ASTNode], List[str], Optional[str]]:
        """
        Parse a Verilog transition condition.
        Returns: (success, ast_node, referenced_identifiers, error_message)
        """
        raw = cond_text.strip().rstrip(";")
        if not raw:
            return False, None, [], "Condition expression cannot be empty"

        # Check for prose before parsing
        prose_err = cls.check_prose(raw)
        if prose_err:
            return False, None, [], prose_err

        # Check for assignment statement confusion in condition (e.g. `state <= IDLE` or `out = 1`)
        if re.search(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<=|=)\s*[^=]", raw):
            # Could be assignment used as condition or <= confusion
            # Check if <= is used as comparison (e.g. `cnt <= 10`)
            # In Verilog, cnt <= 10 is syntactically a valid comparison, but if it has a trailing semicolon or looks like an action:
            if ";" in cond_text or re.search(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]", raw):
                return False, None, [], f"Assignment statement used as transition condition: '{raw}'"

        tokens, err = ExpressionTokenizer.tokenize(raw)
        if err:
            return False, None, [], err

        parser = VerilogPrecedenceParser(tokens, raw)
        ast, err = parser.parse_expression(1)
        if err:
            return False, None, [], err

        if parser.current().type != TokenType.EOF:
            return False, None, [], f"Unexpected trailing tokens '{parser.current().value}' in condition: '{raw}'"

        # Verify all identifiers
        norm_symbols = {s.strip().lower() for s in declared_symbols}
        all_ids = cls.collect_identifiers(ast)
        ref_names = []

        for id_name, _ in all_ids:
            clean_name = id_name.strip()
            lower_name = clean_name.lower()
            if lower_name in VERILOG_KEYWORDS:
                return False, None, [], f"Verilog keyword '{clean_name}' cannot be used as an identifier in condition"
            if lower_name not in norm_symbols:
                return False, None, [], f"Unknown identifier '{clean_name}' referenced in condition is not a declared port, register, parameter, or state"
            ref_names.append(clean_name)

        return True, ast, ref_names, None

    @classmethod
    def parse_action(
        cls,
        action_text: str,
        declared_symbols: Set[str],
        writable_symbols: Optional[Set[str]] = None
    ) -> Tuple[bool, List[ActionAssignmentNode], List[str], Optional[str]]:
        """
        Parse Verilog transition action statements (e.g. `out <= 1; cnt <= cnt + 1;`).
        Returns: (success, list_of_action_nodes, referenced_identifiers, error_message)
        """
        raw = action_text.strip()
        if not raw:
            return True, [], [], None

        prose_err = cls.check_prose(raw)
        if prose_err:
            return False, [], [], prose_err

        # Split multiple statements separated by ';'
        raw_stmts = [s.strip() for s in raw.split(";") if s.strip()]
        action_nodes: List[ActionAssignmentNode] = []
        all_refs: List[str] = []
        norm_symbols = {s.strip().lower() for s in declared_symbols}
        norm_writable = {s.strip().lower() for s in writable_symbols} if writable_symbols else norm_symbols

        for stmt in raw_stmts:
            # Match target [select] op expr
            assign_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_$]*(?:\[[^\]]+\])?)\s*(<=|=)\s*(.+)$", stmt)
            if not assign_match:
                return False, [], [], f"Invalid action statement '{stmt}'. Expected 'target <= expr;' or 'target = expr;'"

            lhs_str = assign_match.group(1).strip()
            op = assign_match.group(2).strip()
            rhs_str = assign_match.group(3).strip()

            # Parse LHS identifier
            lhs_tokens, err = ExpressionTokenizer.tokenize(lhs_str)
            if err:
                return False, [], [], f"Failed to parse action LHS '{lhs_str}': {err}"
            lhs_parser = VerilogPrecedenceParser(lhs_tokens, lhs_str)
            lhs_ast, err = lhs_parser.parse_primary()
            if err or not isinstance(lhs_ast, IdentifierNode):
                return False, [], [], f"Action LHS must be an identifier, got '{lhs_str}'"

            # Check LHS target
            lhs_name = lhs_ast.name.lower()
            if lhs_name in VERILOG_KEYWORDS:
                return False, [], [], f"Verilog keyword '{lhs_ast.name}' cannot be assignment target"
            if lhs_name not in norm_symbols:
                return False, [], [], f"Unknown identifier '{lhs_ast.name}' as assignment target"
            if norm_writable and lhs_name not in norm_writable:
                return False, [], [], f"Identifier '{lhs_ast.name}' is an input port and cannot be assigned in an action"

            # Parse RHS expression
            rhs_tokens, err = ExpressionTokenizer.tokenize(rhs_str)
            if err:
                return False, [], [], f"Failed to parse action RHS '{rhs_str}': {err}"
            rhs_parser = VerilogPrecedenceParser(rhs_tokens, rhs_str)
            rhs_ast, err = rhs_parser.parse_expression(1)
            if err:
                return False, [], [], f"Failed to parse action RHS expression '{rhs_str}': {err}"
            if rhs_parser.current().type != TokenType.EOF:
                return False, [], [], f"Trailing tokens in action RHS '{rhs_str}'"

            # Check RHS identifiers
            rhs_ids = cls.collect_identifiers(rhs_ast)
            for id_n, _ in rhs_ids:
                l_id = id_n.lower()
                if l_id in VERILOG_KEYWORDS:
                    return False, [], [], f"Verilog keyword '{id_n}' cannot be used in action expression"
                if l_id not in norm_symbols:
                    return False, [], [], f"Unknown identifier '{id_n}' in action expression '{rhs_str}'"
                all_refs.append(id_n)

            all_refs.append(lhs_ast.name)
            action_nodes.append(ActionAssignmentNode(lhs_ast, op, rhs_ast))

        return True, action_nodes, all_refs, None


def check_active_low_assertion(
    condition_text: str,
    declared_signals: List[Union[Dict[str, Any], Any]]
) -> Optional[Dict[str, Any]]:
    """
    Check if an active-low signal (_n, _b, _l) is used with prose assertion/high.
    Phase 4 Rule: Do not auto-interpret 'X asserted/high' for active-low names (_n, _b, _l):
    raise a gap instead.
    """
    cond_lower = condition_text.lower()
    for sig in declared_signals:
        s_name = sig.get("name", "") if isinstance(sig, dict) else getattr(sig, "name", "")
        if not s_name:
            continue
        s_lower = s_name.lower()
        is_active_low = s_lower.endswith(("_n", "_b", "_l")) or (
            isinstance(sig, dict) and "low" in str(sig.get("description", "")).lower()
        )
        if is_active_low:
            pattern = rf"\b{re.escape(s_lower)}\b.*?\b(asserted|deasserted|high|low)\b|\b(asserted|deasserted|high|low)\b.*?\b{re.escape(s_lower)}\b"
            if re.search(pattern, cond_lower):
                return {
                    "code": "GAP_ACTIVE_LOW_ASSERTION_AMBIGUOUS",
                    "category": "SIGNAL",
                    "classification": "AMBIGUOUS",
                    "badge": "🔴 [GAP_ACTIVE_LOW_ASSERTION_AMBIGUOUS]",
                    "is_blocking": True,
                    "issue": f"Active-low signal '{s_name}' is used with ambiguous polarity prose in '{condition_text}'. Auto-interpretation is forbidden.",
                    "required_information": f"Explicitly specify whether '{s_name}' asserted corresponds to logic 0 or logic 1.",
                    "evidence": condition_text
                }
    return None


def check_comparison_assignment_confusion(
    condition_text: str,
    declared_states: Set[str]
) -> Optional[str]:
    """
    Detect assignment/comparison confusion such as 'a <= b' where an assignment
    statement was mistakenly used as a transition condition.
    """
    raw = condition_text.strip().rstrip(";")
    norm_states = {s.strip().lower() for s in declared_states}

    # Case 1: `identifier <= STATE_NAME` in condition
    m_assign_state = re.match(r"^([a-zA-Z_][a-zA-Z0-9_$]*)\s*<=\s*([a-zA-Z_][a-zA-Z0-9_$]*)$", raw)
    if m_assign_state:
        rhs = m_assign_state.group(2).lower()
        if rhs in norm_states:
            return f"Comparison/assignment confusion: '{raw}' is a non-blocking state assignment, not a boolean transition condition. Use '==' for comparison or place assignment in transition actions."

    # Case 2: `identifier = expr` (single '=') in condition
    if re.search(r"(?<![<>=!~&|^+*/%-])=(?![=])", raw):
        return f"Comparison/assignment confusion: single '=' found in condition '{raw}'. Use '==' for equality comparison or '<=' for action assignment."

    return None


def compute_priority_masked_conditions(
    transitions_by_state: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Mask transition actions by priority:
    An action on transition k must fire only when its source state matches,
    its condition is true, and no earlier transition from that state fires.
    """
    masked_by_state: Dict[str, List[Dict[str, Any]]] = {}
    for state, trans_list in transitions_by_state.items():
        masked_list = []
        prior_conditions: List[str] = []
        for k, tr in enumerate(trans_list):
            raw_cond = tr.get("condition")
            if isinstance(raw_cond, dict):
                cond_str = raw_cond.get("expression", "1'b1") or "1'b1"
            else:
                cond_str = str(raw_cond) if raw_cond is not None else "1'b1"

            if k == 0:
                p_mask = f"({cond_str})"
            else:
                not_priors = " && ".join(f"!({pc})" for pc in prior_conditions if pc not in ("1'b1", "1"))
                if not_priors:
                    p_mask = f"({not_priors}) && ({cond_str})"
                else:
                    p_mask = f"({cond_str})"

            prior_conditions.append(cond_str)
            tr_copy = dict(tr)
            tr_copy["priority_index"] = k
            tr_copy["priority_mask"] = p_mask
            masked_list.append(tr_copy)
        masked_by_state[state] = masked_list
    return masked_by_state

