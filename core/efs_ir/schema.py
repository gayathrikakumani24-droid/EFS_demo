"""
Canonical EFS Hardware Intermediate Representation (EFS IR) JSON Schema.

Defines the formal schema and deterministic structural validator for EFS IR v1.0,
ensuring all generated and loaded EFS IR instances adhere to the single source of truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


EFS_IR_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Canonical EFS Hardware Intermediate Representation (EFS IR)",
    "version": "1.0",
    "type": "object",
    "required": [
        "metadata",
        "components",
        "interfaces",
        "signals",
        "clocks",
        "resets",
        "registers",
        "validation",
        "extraction_quality"
    ],
    "properties": {
        "metadata": {
            "type": "object",
            "required": [
                "design_id",
                "canonical_id",
                "display_name",
                "protocol",
                "ir_schema_version",
                "source_documents"
            ],
            "properties": {
                "design_id": {"type": "string", "minLength": 1},
                "design_name": {"type": "string"},
                "canonical_id": {"type": "string", "minLength": 1},
                "display_name": {"type": "string", "minLength": 1},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "project_name": {"type": "string"},
                "protocol": {"type": "string"},
                "protocol_version": {"type": "string"},
                "document_id": {"type": "string"},
                "document_hash": {"type": "string"},
                "ingestion_id": {"type": "string"},
                "generation_run_id": {"type": "string"},
                "source_documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "hash": {"type": "string"},
                            "revision": {"type": "string"},
                            "ingestion_id": {"type": "string"}
                        }
                    }
                },
                "specification_revision": {"type": "string"},
                "ir_schema_version": {"type": "string"},
                "generation_timestamp": {"type": "string"}
            }
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["component_id", "name"],
                "properties": {
                    "component_id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "interfaces": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "interfaces": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["interface_id", "name"],
                "properties": {
                    "interface_id": {"type": "string"},
                    "name": {"type": "string"},
                    "protocol": {"type": "string"},
                    "role": {"type": "string"},
                    "signals": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["signal_id", "name", "direction"],
                "properties": {
                    "signal_id": {"type": "string"},
                    "name": {"type": "string"},
                    "width": {"type": ["string", "integer"]},
                    "direction": {"type": "string", "enum": ["input", "output", "inout"]},
                    "datatype": {"type": "string"},
                    "semantic_role": {"type": "string"}
                }
            }
        },
        "clocks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["signal_id"],
                "properties": {
                    "signal_id": {"type": "string"}
                }
            }
        },
        "resets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["signal_id"],
                "properties": {
                    "signal_id": {"type": "string"}
                }
            }
        },
        "parameters": {"type": "array"},
        "registers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["register_id", "name", "offset"],
                "properties": {
                    "register_id": {"type": "string"},
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                    "offset": {"type": ["string", "null"]},
                    "offset_source": {"type": ["string", "null"]},
                    "width": {"type": ["integer", "null"]},
                    "access": {"type": "string"},
                    "access_type": {"type": "string"},
                    "reset_value": {"type": ["string", "null"]},
                    "fields": {"type": "array"},
                    "description": {"type": "string"}
                }
            }
        },
        "instructions": {"type": "array"},
        "instruction_formats": {"type": "array"},
        "opcodes": {"type": "array"},
        "fsms": {"type": "array"},
        "transactions": {"type": "array"},
        "functional_behavior": {"type": "array"},
        "memory": {"type": "array"},
        "protocol_rules": {"type": "array"},
        "timing_rules": {"type": "array"},
        "timing_requirements": {"type": "array"},
        "performance_requirements": {"type": "array"},
        "errors": {"type": "array"},
        "constraints": {"type": "array"},
        "requirements": {"type": "array"},
        "validation": {
            "type": "object",
            "required": ["status", "summary", "findings"],
            "properties": {
                "status": {"type": "string"},
                "design_id": {"type": "string"},
                "canonical_id": {"type": "string"},
                "generation_run_id": {"type": "string"},
                "document_id": {"type": "string"},
                "ingestion_id": {"type": "string"},
                "summary": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "critical": {"type": "integer"},
                        "warning": {"type": "integer"}
                    }
                },
                "findings": {"type": "array"}
            }
        },
        "resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "resolution_id": {"type": "string"},
                    "conflict_id": {"type": "string"},
                    "target_entity": {"type": "string"},
                    "property": {"type": "string"},
                    "original_value": {},
                    "resolved_value": {},
                    "resolution_source": {"type": "string"},
                    "validation_status": {"type": "string"}
                }
            }
        },
        "examples": {"type": "array"},
        "ambiguities": {"type": "array"},
        "contradictions": {"type": "array"},
        "extraction_quality": {
            "type": "object",
            "required": [
                "extraction_status",
                "source_completeness",
                "ir_completeness"
            ],
            "properties": {
                "extraction_status": {"type": "string"},
                "source_completeness": {"type": "string"},
                "ir_completeness": {"type": "string"},
                "missing_information": {"type": "array"}
            }
        }
    }
}


def validate_efs_ir_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an EFS IR dictionary against the formal JSON schema.
    Uses jsonschema if available and performs deterministic structural checks.
    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["EFS IR root must be a JSON object (dictionary)."]

    # 0. Formal jsonschema validation pass if library is installed
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(EFS_IR_SCHEMA)
        for err in validator.iter_errors(data):
            path_str = " -> ".join(str(p) for p in err.path) or "root"
            errors.append(f"Schema validation error at '{path_str}': {err.message}")
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"Formal schema check exception: {e}")

    # 1. Check required top-level keys
    for req_key in EFS_IR_SCHEMA["required"]:
        if req_key not in data:
            errors.append(f"Missing required top-level property: '{req_key}'")

    # 2. Check metadata
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        errors.append("Property 'metadata' must be an object.")
    else:
        for req_meta in ["design_id", "canonical_id", "display_name", "protocol", "ir_schema_version"]:
            val = meta.get(req_meta)
            if not val or not str(val).strip():
                errors.append(f"Metadata property '{req_meta}' cannot be empty.")

        source_docs = meta.get("source_documents")
        if not isinstance(source_docs, list):
            errors.append("Metadata property 'source_documents' must be a list.")

    # 3. Check signals
    signals = data.get("signals", [])
    if isinstance(signals, list):
        for idx, sig in enumerate(signals):
            if not isinstance(sig, dict):
                errors.append(f"signals[{idx}] must be an object.")
                continue
            if not sig.get("signal_id"):
                errors.append(f"signals[{idx}] missing 'signal_id'.")
            if not sig.get("name"):
                errors.append(f"signals[{idx}] missing 'name'.")
            direction = str(sig.get("direction", "")).lower()
            if direction not in ("input", "output", "inout"):
                errors.append(f"signals[{idx}] ('{sig.get('name')}') invalid direction '{direction}'. Must be input, output, or inout.")
    else:
        errors.append("Property 'signals' must be a list.")

    # 4. Check clocks & resets reference format
    for clk_key in ("clocks", "resets"):
        items = data.get(clk_key, [])
        if isinstance(items, list):
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    if "signal_id" not in item:
                        errors.append(f"{clk_key}[{idx}] missing 'signal_id' reference.")
                elif isinstance(item, str):
                    # string ID is acceptable
                    pass
                else:
                    errors.append(f"{clk_key}[{idx}] must be a reference object containing 'signal_id'.")
        else:
            errors.append(f"Property '{clk_key}' must be a list.")

    # 5. Check registers
    regs = data.get("registers", [])
    if isinstance(regs, list):
        for idx, reg in enumerate(regs):
            if not isinstance(reg, dict):
                errors.append(f"registers[{idx}] must be an object.")
                continue
            if not reg.get("register_id"):
                errors.append(f"registers[{idx}] missing 'register_id'.")
            if not reg.get("name"):
                errors.append(f"registers[{idx}] missing 'name'.")
            if "offset" not in reg:
                errors.append(f"registers[{idx}] ('{reg.get('name')}') missing 'offset'.")
    else:
        errors.append("Property 'registers' must be a list.")

    # 6. Check validation structure
    val = data.get("validation")
    if val is not None and not isinstance(val, dict):
        errors.append("Property 'validation' must be an object.")
    elif isinstance(val, dict):
        if "status" not in val:
            errors.append("Validation property missing 'status'.")
        if "findings" not in val or not isinstance(val["findings"], list):
            errors.append("Validation property missing 'findings' list.")

    # 7. Check extraction quality
    eq = data.get("extraction_quality")
    if eq is not None and not isinstance(eq, dict):
        errors.append("Property 'extraction_quality' must be an object.")
    elif isinstance(eq, dict):
        for eq_key in ("extraction_status", "source_completeness", "ir_completeness"):
            if eq_key not in eq:
                errors.append(f"Extraction quality property missing '{eq_key}'.")

    return (len(errors) == 0, errors)
