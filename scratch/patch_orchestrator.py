with open('hardware_graphrag/core/agents/orchestrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update run_design_flow signature
old_sig1 = """def run_design_flow(
    requirement_text: str,
    target_outputs: List[str],
    max_repair_iterations: int = 2,
    compliance_threshold: int = 90,
    progress_cb: Callable[[str, float], None] = None,
    bypass_conflicts: bool = False,
    user_query: Optional[str] = None
) -> Dict[str, Any]:"""

new_sig1 = """def run_design_flow(
    requirement_text: str,
    target_outputs: List[str],
    max_repair_iterations: int = 2,
    compliance_threshold: int = 90,
    progress_cb: Callable[[str, float], None] = None,
    bypass_conflicts: bool = False,
    user_query: Optional[str] = None,
    generation_mode: str = "HYBRID",
    k_candidates: int = 1,
    allow_partial: bool = True,
    **kwargs: Any
) -> Dict[str, Any]:"""

assert old_sig1 in text, "old_sig1 not found"
text = text.replace(old_sig1, new_sig1, 1)

# 2. Update resume_design_flow_with_resolutions signature
old_sig2 = """def resume_design_flow_with_resolutions(
    design_results: Dict[str, Any],
    approved_resolutions: List[Any],
    target_outputs: List[str] = None,
    progress_cb: Callable[[str, float], None] = None,
    bypass_conflicts: bool = False
) -> Dict[str, Any]:"""

new_sig2 = """def resume_design_flow_with_resolutions(
    design_results: Dict[str, Any],
    approved_resolutions: List[Any],
    target_outputs: List[str] = None,
    progress_cb: Callable[[str, float], None] = None,
    bypass_conflicts: bool = False,
    user_query: Optional[str] = None,
    generation_mode: str = "HYBRID",
    k_candidates: int = 1,
    allow_partial: bool = True,
    **kwargs: Any
) -> Dict[str, Any]:"""

assert old_sig2 in text, "old_sig2 not found"
text = text.replace(old_sig2, new_sig2, 1)

# 3. Replace proposal generation with batch proposal generation
old_prop1 = """            proposals_map = {}
            for conf in unresolved_blocking:
                proposals_map[conf.conflict_id] = ConflictResolutionEngine.generate_resolution_proposals(conf, efs_ir)"""

new_prop1 = """            proposals_map = ConflictResolutionEngine.generate_batch_resolution_proposals(unresolved_blocking, efs_ir)"""

assert old_prop1 in text, "old_prop1 not found"
text = text.replace(old_prop1, new_prop1, 1)

old_prop2 = """        proposals_map = {}
        for conf in unresolved_blocking:
            proposals_map[conf.conflict_id] = ConflictResolutionEngine.generate_resolution_proposals(conf, resolved_ir)"""

new_prop2 = """        proposals_map = ConflictResolutionEngine.generate_batch_resolution_proposals(unresolved_blocking, resolved_ir)"""

assert old_prop2 in text, "old_prop2 not found"
text = text.replace(old_prop2, new_prop2, 1)

with open('hardware_graphrag/core/agents/orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated orchestrator.py successfully!")

