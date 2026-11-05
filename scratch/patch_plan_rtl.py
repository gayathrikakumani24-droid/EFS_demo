with open('hardware_graphrag/core/agents/rtl_agent.py', 'rb') as f:
    content = f.read()

target1 = b'def plan_rtl(design_spec: str, target: Any = "Verilog", efs_ir: Optional[EFSIR] = None, context_pack: Optional[Any] = None) -> Dict[str, Any]:'
replacement1 = b'''def plan_rtl(
    design_spec: str,
    target: Any = "Verilog",
    efs_ir: Optional[EFSIR] = None,
    context_pack: Optional[Any] = None,
    generation_mode: str = "HYBRID",
    **kwargs: Any
) -> Dict[str, Any]:'''

assert target1 in content, "target1 not found"
content = content.replace(target1, replacement1, 1)

target2 = b'rtl_plan = plan_rtl(design_spec, target_lang, efs_ir=efs_ir, context_pack=context_pack)'
replacement2 = b'rtl_plan = plan_rtl(design_spec, target_lang, efs_ir=efs_ir, context_pack=context_pack, generation_mode=generation_mode)'

assert target2 in content, "target2 not found"
content = content.replace(target2, replacement2, 1)

with open('hardware_graphrag/core/agents/rtl_agent.py', 'wb') as f:
    f.write(content)

print("Patched rtl_agent.py successfully")

