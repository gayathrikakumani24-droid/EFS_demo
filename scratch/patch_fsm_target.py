with open('hardware_graphrag/core/agents/fsm_inference_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        # Detect pure combinational
        if arch_res.primary_class == ArchitectureClass.COMBINATIONAL or (not arch_res.has_clock and not arch_res.has_reset and not arch_res.requires_fsm):
            logger.info("FSM Case 4: Pure combinational circuit - FSM is unnecessary.")
            return FSMInferenceResult(
                case=FSMHandlingCase.FSM_UNNECESSARY,
                fsm=None,
                requires_hitl_approval=False,
                inference_notes="Pure combinational logic does not require an FSM."
            )

        # Detect simple register/counter datapath without multi-state controller
        is_simple_pipeline_or_counter = (
            arch_res.primary_class in (ArchitectureClass.DATAPATH, ArchitectureClass.SEQUENTIAL, ArchitectureClass.ARITHMETIC_PROCESSING_UNIT)
            and not arch_res.requires_fsm
            and not re.search(r"\\b(state\\s+machine|fsm|states:|transition)\\b", text_lower)
        )
        if is_simple_pipeline_or_counter:"""

replacement = """        has_fsm_target = bool(
            (context_pack and isinstance(context_pack, dict) and (context_pack.get("target_fsm") or context_pack.get("fsm")))
            or (context_pack and hasattr(context_pack, "target_fsm") and getattr(context_pack, "target_fsm", None))
            or (efs_ir and getattr(efs_ir, "fsms", None) and len(efs_ir.fsms) > 0)
        )

        # Detect pure combinational
        if not has_fsm_target and (arch_res.primary_class == ArchitectureClass.COMBINATIONAL or (not arch_res.has_clock and not arch_res.has_reset and not arch_res.requires_fsm)):
            logger.info("FSM Case 4: Pure combinational circuit - FSM is unnecessary.")
            return FSMInferenceResult(
                case=FSMHandlingCase.FSM_UNNECESSARY,
                fsm=None,
                requires_hitl_approval=False,
                inference_notes="Pure combinational logic does not require an FSM."
            )

        # Detect simple register/counter datapath without multi-state controller
        is_simple_pipeline_or_counter = (
            arch_res.primary_class in (ArchitectureClass.DATAPATH, ArchitectureClass.SEQUENTIAL, ArchitectureClass.ARITHMETIC_PROCESSING_UNIT)
            and not arch_res.requires_fsm
            and not re.search(r"\\b(state\\s+machine|fsm|states:|transition)\\b", text_lower)
        )
        if is_simple_pipeline_or_counter and not has_fsm_target:"""

normalized = text.replace('\r\n', '\n')
if target in normalized:
    new_text = normalized.replace(target, replacement, 1)
    with open('hardware_graphrag/core/agents/fsm_inference_engine.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('SUCCESS')
else:
    print('TARGET NOT FOUND')
