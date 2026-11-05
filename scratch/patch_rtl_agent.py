with open('hardware_graphrag/core/agents/rtl_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''                    fsm_transitions.append({
                        "from": from_st,
                        "to": to_st,
                        "source_ref": cond_source_evid or f"{from_st} -> {to_st}: {cond_expr}",
                        "condition": {


    scoped_signals = []'''

replacement = '''                    fsm_transitions.append({
                        "from": from_st,
                        "to": to_st,
                        "source_ref": cond_source_evid or f"{from_st} -> {to_st}: {cond_expr}",
                        "condition": {
                            "expression": "1'b1" if is_dwell else (cond_expr if (not is_vague or is_resolved) else None),
                            "signals": [],
                            "registers": [],
                            "protocol_events": [],
                            "source": "EFS",
                            "source_evidence": cond_source_evid,
                            "is_fully_defined": is_dwell or (not is_vague)
                        }
                    })

    # Detect reset polarity without guessing
    rst_port_obj = next((p for p in ports if any(k in str(p.get("name", "")).lower() for k in ("rst", "reset"))), None)
    rst_pol_detected, _ = detect_reset_polarity(
        rst_port_obj.get("name") if rst_port_obj else "rst_n",
        description=rst_port_obj.get("description") if rst_port_obj else None,
        efs_ir=efs_ir
    )

    plan = {
        "module_name": target_module_name if target_module_name not in ("target_module", "hardware_module", "custom_module") else spec_parsed.get("module_name", target_module_name),
        "specification_status": spec_parsed.get("specification_status", {
            "status": "COMPLETE",
            "blocking_issues": []
        }),
        "ports": ports,
        "parameters": spec_parsed.get("parameters", []),
        "internal_registers": internal_regs,
        "fsm": {
            "states": fsm_states,
            "initial_state": fsm_states[0] if fsm_states else None,
            "transitions": fsm_transitions
        },
        "datapath_strategy": "Direct state machine & register transfer logic",
        "reset_polarity": rst_pol_detected or "active_low",
        "design_spec": design_spec
    }

    val = validate_rtl_plan(plan, efs_ir=efs_ir, context_pack=context_pack, design_spec=design_spec)
    plan["specification_status"] = {
        "status": val["status"],
        "blocking_issues": val["issues"]
    }
    return plan


def validate_rtl_plan(
    rtl_plan: Dict[str, Any],
    efs_ir: Optional[EFSIR] = None,
    context_pack: Optional[Any] = None,
    design_spec: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage B Validation: Verifies that the generated RTL plan faithfully implements the extracted EFSIR or scoped ContextPack.
    Compares: RTL Plan <-> EFSIR / ContextPack (NOT against generic assumed hardware templates).
    """
    logger.info("Validating RTL plan against authoritative EFS IR or scoped ContextPack...")
    all_issues = []

    # Version and Source Consistency Logging
    model_ver = getattr(efs_ir.metadata, "model_version", 1) if (efs_ir and hasattr(efs_ir, "metadata")) else 1
    res_ver = getattr(efs_ir.metadata, "resolution_version", 0) if (efs_ir and hasattr(efs_ir, "metadata")) else 0
    ctx_ver = getattr(context_pack, "context_version", model_ver) if context_pack else model_ver
    val_source = "resolved_efs_ir" if res_ver > 0 else ("context_pack" if context_pack else "efs_ir")

    logger.info(
        f"MODEL_VERSION={model_ver} "
        f"EFS_IR_VERSION={model_ver} "
        f"GENERATION_CONTEXT_VERSION={ctx_ver} "
        f"RTL_VALIDATION_SOURCE={val_source}"
    )

    if not isinstance(rtl_plan, dict):
        return {
            "valid": False,
            "status": "INCOMPLETE",
            "issues": [{
                "code": "PLAN_SCHEMA_VIOLATION",
                "category": "INTERFACE",
                "classification": "PLAN_MISSING",
                "badge": "🔴 [PLAN_MISSING]",
                "is_blocking": True,
                "issue": "Structured RTL Plan is not a valid JSON object.",
                "required_information": "Provide a valid structured plan.",
                "evidence": str(rtl_plan)
            }]
        }

    # 0. Plan Schema Validation (Reject/Repair on schema errors instead of tolerating missing keys)
    is_schema_valid, _, schema_issues = validate_plan_schema(rtl_plan)
    if not is_schema_valid:
        all_issues.extend(schema_issues)

    # 0b. Source Ref Traceability Validation (Prevent invented behaviour by checking verifiable quotes/IDs)
    spec_str = design_spec or rtl_plan.get("design_spec") or ""
    if not spec_str and context_pack and hasattr(context_pack, "design_spec"):
        spec_str = getattr(context_pack, "design_spec", "")
    if spec_str or efs_ir:
        source_ref_issues = validate_plan_source_refs(rtl_plan, design_spec=spec_str, efs_ir=efs_ir, context_pack=context_pack)
        all_issues.extend(source_ref_issues)

    scoped_signals = []'''

if target in content:
    new_content = content.replace(target, replacement, 1)
    with open('hardware_graphrag/core/agents/rtl_agent.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS")
else:
    print("TARGET NOT FOUND")

