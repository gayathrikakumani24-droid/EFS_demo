import re

# 1. Fix spec_classifier.py
p1 = "hardware_graphrag/core/agents/spec_classifier.py"
with open(p1, "r", encoding="utf-8") as f:
    t1 = f.read()
t1 = t1.replace(
    r"\b(state\s+machine|states:\s*[a-z0-9_,\s]+|state\s+s_[a-z0-9_]+|transition\s+to\s+s_[a-z0-9_]+|dwells\s+in\s+s_[a-z0-9_]+|3-state|sequence\s+controller)\b",
    r"\b(state\s+machine|states?[:\s]+[a-z0-9_,\s]+|state\s+s_[a-z0-9_]+|transition\s+to\s+s_[a-z0-9_]+|dwells\s+in\s+s_[a-z0-9_]+|3-state|sequence\s+controller)\b"
)
with open(p1, "w", encoding="utf-8") as f:
    f.write(t1)
print("Updated spec_classifier.py")

# 2. Fix fsm_inference_engine.py
p2 = "hardware_graphrag/core/agents/fsm_inference_engine.py"
with open(p2, "r", encoding="utf-8") as f:
    t2 = f.read()
t2 = t2.replace(
    r"and not re.search(r\"\b(state\s+machine|fsm|states:|transition)\b\", text_lower)",
    r"and not re.search(r\"\b(state\s+machine|fsm|states?[:\s]|transition)\b\", text_lower)"
)
with open(p2, "w", encoding="utf-8") as f:
    f.write(t2)
print("Updated fsm_inference_engine.py")

# 3. Fix rtl_plan_validator.py
p3 = "hardware_graphrag/core/agents/rtl_plan_validator.py"
with open(p3, "r", encoding="utf-8") as f:
    t3 = f.read()
t3 = t3.replace(
    """        out_drivers = plan_dict.get("output_driving_strategy") or plan_dict.get("output_assignments") or {}
        if out_ports and out_drivers:""",
    """        has_strat_field = ("output_driving_strategy" in plan_dict or "output_assignments" in plan_dict)
        out_drivers = plan_dict.get("output_driving_strategy") if "output_driving_strategy" in plan_dict else plan_dict.get("output_assignments")
        if out_ports and has_strat_field:"""
)
with open(p3, "w", encoding="utf-8") as f:
    f.write(t3)
print("Updated rtl_plan_validator.py")

# 4. Fix hitl_plan_manager.py
p4 = "hardware_graphrag/core/verification/hitl_plan_manager.py"
with open(p4, "r", encoding="utf-8") as f:
    t4 = f.read()
t4 = t4.replace(
    """        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=conflict_id,
            conflict_type=getattr(matched_conflict, "type", "SPEC_CONFLICT") if matched_conflict else "SPEC_CONFLICT",
            target_entity=target_entity or (matched_conflict.entities[0] if matched_conflict and getattr(matched_conflict, "entities", None) else conflict_id),
            target_property=target_property or getattr(matched_conflict, "category", "general"),
            resolved_value=approved_value,
            approval_status="APPROVED",
            resolution_source="HUMAN",
            explanation=explanation,
            affected_requirements=[conflict_id, target_entity] if target_entity else [conflict_id],
        )""",
    """        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=conflict_id,
            target_entity=target_entity or (matched_conflict.entities[0] if matched_conflict and getattr(matched_conflict, "entities", None) else conflict_id),
            property=target_property or getattr(matched_conflict, "category", "general"),
            resolved_value=approved_value,
            approval_status="APPROVED",
            resolution_source="human",
            rationale=explanation,
            reason=explanation,
            affected_requirements=[conflict_id, target_entity] if target_entity else [conflict_id],
        )"""
)
t4 = t4.replace(
    """        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=conflict_id,
            conflict_type="SPEC_CONFLICT",
            target_entity=conflict_id,
            approval_status="REJECTED",
            resolution_source="HUMAN",
            explanation=reason,
            affected_requirements=[conflict_id]
        )""",
    """        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=conflict_id,
            target_entity=conflict_id,
            approval_status="REJECTED",
            resolution_source="human",
            rationale=reason,
            reason=reason,
            affected_requirements=[conflict_id]
        )"""
)
with open(p4, "w", encoding="utf-8") as f:
    f.write(t4)
print("Updated hitl_plan_manager.py")

# 5. Fix test_subprompt1_rtl_planning_and_validation.py
p5 = "hardware_graphrag/tests/test_subprompt1_rtl_planning_and_validation.py"
with open(p5, "r", encoding="utf-8") as f:
    t5 = f.read()
t5 = t5.replace(
    'EFSOpcode(name="OP_ADD", binary_encoding="4\'b0001")',
    'EFSOpcode(mnemonic="OP_ADD", binary_encoding="4\'b0001")'
)
t5 = t5.replace(
    'EFSOpcode(name="OP_SUB", binary_encoding="4\'b0001")',
    'EFSOpcode(mnemonic="OP_SUB", binary_encoding="4\'b0001")'
)
with open(p5, "w", encoding="utf-8") as f:
    f.write(t5)
print("Updated test_subprompt1_rtl_planning_and_validation.py")

