"""
EFS IR Builder.

Builds a canonical EFSIR instance from the Requirement Model, Architecture Plan,
retrieved protocol rules, and the ingestion registry (to establish source traceability).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from core.registry import get_registry
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSSourceDocument, EFSTraceability, EFSComponent, EFSInterface, EFSSignal,
    EFSRegister, EFSRegisterField, EFSFSM, EFSState, EFSTransition,
    EFSConstraint, EFSTimingRule, EFSProtocolRule, EFSFlow, EFSFlowStep,
    EFSAssertionIntent, EFSCoverageIntent, new_efs_id,
    EFSInstruction, EFSInstructionField, EFSOpcode, EFSMemoryRegion,
    EFSDataStructure, EFSError, EFSPerformanceRequirement, EFSConflict,
    EFSExtractionQuality, EFSResolutionCandidate, EFSTransaction
)
from core.efs_ir.validation_rules import (
    is_valid_register_candidate,
    normalize_hardware_offset,
    is_explicit_address,
)
from utils.logger import get_logger

logger = get_logger("efs_ir.builder")


class EFSIRBuilder:
    """Builder class to assemble the EFS IR from pipeline and design flow artifacts."""

    def __init__(
        self,
        requirement_model: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
        protocol_rules: Optional[List[Dict[str, Any]]] = None,
        design_spec: str = "",
        req_model: Optional[Dict[str, Any]] = None,
        generation_run_id: Optional[str] = None
    ):
        self.req_model = requirement_model or req_model or {}
        self.plan = plan or {}
        self.protocol_rules = protocol_rules or []
        self.design_spec = design_spec
        self.generation_run_id = generation_run_id or self.req_model.get("generation_run_id")
        
        self.registry = get_registry()
        self.ir = EFSIR()
        
        # Helper mappings for ID resolution
        self.name_to_component_id: Dict[str, str] = {}
        self.name_to_interface_id: Dict[str, str] = {}
        self.name_to_signal_id: Dict[str, str] = {}
        self.name_to_register_id: Dict[str, str] = {}

    def _find_entity_traceability(self, entity_name: str) -> EFSTraceability:
        """Scan the pipeline registry to find source document / chunk traceability for a name."""
        entity_name_lower = entity_name.lower().replace("_", " ")
        for doc_id, entities in self.registry.entities.items():
            doc_obj = self.registry.documents.get(doc_id)
            doc_name = doc_obj.filename if doc_obj else "specification.docx"
            doc_hash = getattr(doc_obj, "document_hash", "") if doc_obj else ""
            ingest_id = getattr(doc_obj, "ingestion_id", "") if doc_obj else ""
            for ent in entities:
                ent_name_lower = ent.name.lower().replace("_", " ")
                # Match canonical name or aliases
                if ent_name_lower == entity_name_lower or any(a.lower().replace("_", " ") == entity_name_lower for a in ent.aliases):
                    return EFSTraceability(
                        doc_id=doc_id,
                        source_document_id=doc_id,
                        source_document=doc_name,
                        document_hash=doc_hash,
                        ingestion_id=ingest_id,
                        chunk_id=ent.chunk_id,
                        page=ent.page,
                        chapter=ent.chapter,
                        section=ent.section,
                        original_text=ent.original_text,
                        source_text=ent.original_text,
                        source_entity_id=ent.entity_id,
                        extraction_method="pipeline_normalized",
                        confidence=0.95
                    )
        
        # Check if we have a default registered document
        first_doc = next(iter(self.registry.documents.values()), None) if self.registry.documents else None
        if first_doc:
            return EFSTraceability(
                doc_id=first_doc.doc_id,
                source_document_id=first_doc.doc_id,
                source_document=first_doc.filename,
                document_hash=first_doc.document_hash,
                ingestion_id=first_doc.ingestion_id,
                chunk_id="SPEC_EXTRACTION",
                original_text=f"Specification entry for '{entity_name}'",
                source_text=f"Specification entry for '{entity_name}'",
                extraction_method="heuristic_extracted",
                confidence=0.85
            )
            
        return EFSTraceability(
            doc_id="DOC_user_spec",
            source_document_id="DOC_user_spec",
            source_document="user_specification",
            chunk_id="USER_SPEC_TEXT",
            original_text=f"Specification entry for '{entity_name}'",
            source_text=f"Specification entry for '{entity_name}'",
            extraction_method="heuristic_extracted",
            confidence=0.85
        )

    def _find_relationship_traceability(self, source: str, target: str) -> EFSTraceability:
        """Scan the pipeline registry for relationships between two entities for traceability."""
        src_lower = source.lower()
        tgt_lower = target.lower()
        for doc_id, rels in self.registry.relationships.items():
            doc_obj = self.registry.documents.get(doc_id)
            doc_name = doc_obj.filename if doc_obj else "specification.docx"
            doc_hash = getattr(doc_obj, "document_hash", "") if doc_obj else ""
            ingest_id = getattr(doc_obj, "ingestion_id", "") if doc_obj else ""
            for rel in rels:
                if rel.source.lower() == src_lower and rel.target.lower() == tgt_lower:
                    return EFSTraceability(
                        doc_id=doc_id,
                        source_document_id=doc_id,
                        source_document=doc_name,
                        document_hash=doc_hash,
                        ingestion_id=ingest_id,
                        chunk_id=rel.chunk_id,
                        source_relationship_id=rel.rel_id,
                        original_text=rel.evidence,
                        source_text=rel.evidence,
                        extraction_method="pipeline_relation",
                        confidence=0.9
                    )
        return EFSTraceability(extraction_method="unknown", confidence=None)

    def build(self) -> EFSIR:
        """Execute the end-to-end building sequence."""
        self._build_metadata()
        self._build_components()
        self._build_interfaces_and_signals()
        self._build_registers()
        self._build_instructions()
        self._build_opcodes()
        self._build_memory_regions()
        self._build_data_structures()
        self._build_errors()
        self._build_performance_requirements()
        self._build_fsms()
        self._build_protocol_rules_and_constraints()
        self._build_flows()
        self._build_transactions()
        self._build_assertion_intent()
        self._build_coverage_intent()
        self._build_conflicts()
        self._build_extraction_quality()
        return self.ir

    def _build_metadata(self):
        """Build design metadata with canonical ID, display name, and aliases."""
        proto = "generic"
        if self.req_model.get("protocol_references"):
            proto = self.req_model["protocol_references"][0]
            
        source_docs: List[EFSSourceDocument] = []
        seen_doc_ids = set()
        doc_id = ""
        doc_hash = ""
        ingestion_id = ""
        
        if self.registry.documents:
            # If req_model specifies a document_id, prioritize it first
            target_doc_id = self.req_model.get("document_id")
            sorted_docs = sorted(
                self.registry.documents.values(),
                key=lambda d: 0 if (target_doc_id and d.doc_id == target_doc_id) else 1
            )
            for d in sorted_docs:
                if d.doc_id not in seen_doc_ids:
                    seen_doc_ids.add(d.doc_id)
                    source_docs.append(EFSSourceDocument(
                        document_id=d.doc_id,
                        name=d.filename,
                        type=d.file_type,
                        hash=getattr(d, "document_hash", ""),
                        revision="1.0",
                        ingestion_id=getattr(d, "ingestion_id", "")
                    ))
            if source_docs:
                doc_id = source_docs[0].document_id
                doc_hash = source_docs[0].hash
                ingestion_id = source_docs[0].ingestion_id

        if not source_docs:
            doc_name = self.req_model.get("source_document") or self.req_model.get("document_name") or "specification.docx"
            doc_id = self.req_model.get("document_id") or new_efs_id("doc")
            new_doc = EFSSourceDocument(
                document_id=doc_id,
                name=doc_name,
                type="docx" if doc_name.endswith(".docx") else "text",
                revision="1.0"
            )
            source_docs.append(new_doc)
            doc_hash = new_doc.hash
            ingestion_id = new_doc.ingestion_id

        run_id = self.generation_run_id or new_efs_id("run")
            
        # Extract display name from spec text header, req_model, or plan
        def _clean_title(raw: str) -> str:
            if not raw:
                return ""
            s = str(raw).strip()
            parts = s.split()
            while parts and parts[0].lower() in ("the", "a", "an"):
                parts = parts[1:]
            return " ".join(parts).strip()

        raw_display_name = ""
        for k in ("design_spec_title", "title", "display_name", "design_name", "name"):
            if self.req_model.get(k):
                candidate = _clean_title(str(self.req_model[k]))
                if candidate and candidate.lower() not in ("target_module", "target module", "unnamed_design", "target_component"):
                    raw_display_name = candidate
                    break

        if not raw_display_name and self.plan:
            for k in ("target", "target_component", "design_name", "title"):
                if self.plan.get(k):
                    candidate = _clean_title(str(self.plan[k]))
                    if candidate and candidate.lower() not in ("target_module", "target module", "unnamed_design", "target_component"):
                        raw_display_name = candidate
                        break

        if not raw_display_name and self.design_spec:
            name_match = re.search(r"#\s*Hardware Design Specification:\s*([A-Za-z0-9_\-\s]+)", self.design_spec)
            if not name_match:
                name_match = re.search(r"(?:module|design|specification:)\s+([A-Za-z0-9_\-\s]+?)(?:\s+with|\s*\n|$)", self.design_spec, re.IGNORECASE)
            if name_match:
                candidate = _clean_title(name_match.group(1))
                if candidate and candidate.lower() not in ("target_module", "target module", "specification", "table of contents"):
                    raw_display_name = candidate

        if not raw_display_name and source_docs:
            doc_fn = source_docs[0].name
            cand_fn = _clean_title(doc_fn.rsplit(".", 1)[0].replace("_", " ").title())
            if cand_fn and cand_fn.lower() not in ("specification", "spec", "document"):
                raw_display_name = cand_fn

        if not raw_display_name and self.req_model.get("modules"):
            first_m = self.req_model["modules"][0]
            m_name = first_m.get("name") if isinstance(first_m, dict) else str(first_m)
            cand_m = _clean_title(m_name)
            if cand_m and cand_m.lower() not in ("target_module", "target module"):
                raw_display_name = cand_m

        if not raw_display_name:
            arch_title = self.plan.get("system_architecture", "")
            words_arch = [w for w in arch_title.split() if w.lower() not in ("the", "a", "an", "is", "of", "and")]
            if words_arch:
                raw_display_name = words_arch[0].replace(",", "").replace(".", "")

        if not raw_display_name or raw_display_name.lower() in ("the", "a", "an", "target_module", "target module", ""):
            logger.warning("DIAGNOSTIC: Cannot resolve specific design name from metadata, spec, plan, or documents.")
            raw_display_name = "Hardware_Module"

        display_name = raw_display_name
        design_name = re.sub(r"[^a-zA-Z0-9_]", "_", display_name).strip("_").lower() or "hardware_module"
        
        design_id = self.req_model.get("design_id") or new_efs_id("dsn")
        clean_dsn_suffix = design_id.replace("DSN_", "")
        canonical_id = f"EFS_{clean_dsn_suffix}"
        
        # Build comprehensive alias list
        aliases = []
        for name_variant in (display_name, design_name, canonical_id):
            if name_variant and name_variant not in aliases:
                aliases.append(name_variant)
                
        # Add camelCase / PascalCase / snake_case variants
        words = [w for w in re.split(r"[^a-zA-Z0-9]+", display_name) if w]
        if words:
            pascal_name = "".join(w.capitalize() for w in words)
            camel_name = words[0].lower() + "".join(w.capitalize() for w in words[1:])
            snake_name = "_".join(w.lower() for w in words)
            flat_lower = "".join(w.lower() for w in words)
            for v in (pascal_name, camel_name, snake_name, flat_lower):
                if v and v not in aliases:
                    aliases.append(v)
                    
            # Non-prefix and inflection alias variants
            core_words = [w for w in words if w.lower() not in ("simple", "generic", "custom", "top", "main")]
            if core_words:
                core_pascal = "".join(w.capitalize() for w in core_words)
                for v in (core_pascal, core_pascal.replace("Register", "Registered"), core_pascal.replace("Registered", "Register")):
                    if v and v not in aliases:
                        aliases.append(v)

        self.ir.metadata = EFSMetadata(
            design_id=design_id,
            design_name=design_name,
            canonical_id=canonical_id,
            display_name=display_name,
            aliases=aliases,
            project_name=self.req_model.get("project_name", "default_project"),
            protocol=proto,
            protocol_version=self.req_model.get("protocol_version", "1.0"),
            document_id=doc_id,
            document_hash=doc_hash,
            ingestion_id=ingestion_id,
            generation_run_id=run_id,
            source_documents=source_docs,
            specification_revision="1.0",
            ir_schema_version="1.0"
        )

    def _build_components(self):
        """Map architecture plan submodules into EFSComponents and register top-level design target."""
        # 1. Add submodules from plan
        for sub in self.plan.get("submodules", []):
            comp_id = new_efs_id("comp")
            name = sub.get("name", "Unnamed_Submodule")
            comp = EFSComponent(
                component_id=comp_id,
                name=name,
                type=name.lower(),
                description=sub.get("purpose", ""),
                traceability=self._find_entity_traceability(name)
            )
            self.ir.components.append(comp)
            self.name_to_component_id[name] = comp_id
            
        # 2. Add memories from requirement model as components
        for mem in self.req_model.get("memories", []):
            comp_id = new_efs_id("comp")
            name = mem.get("name", "FIFO_Buffer")
            comp = EFSComponent(
                component_id=comp_id,
                name=name,
                type="memory",
                description=f"{mem.get('description', '')} (Depth: {mem.get('depth', 0)}, Width: {mem.get('width', 0)})",
                traceability=self._find_entity_traceability(name)
            )
            self.ir.components.append(comp)
            self.name_to_component_id[name] = comp_id

        # 2b. Add discovered architectural component entities from registry and requirement model
        comp_types = {"component", "module", "subsystem", "block", "unit", "functional_unit", "processing_element", "custom_entity", "core_component", "architectural_component"}
        
        discovered_entities = []
        for ent in self.registry.all_entities():
            if getattr(ent, "entity_type", "").lower() in comp_types:
                discovered_entities.append((ent.name, getattr(ent, "entity_type", "component"), getattr(ent, "description", "")))
        for ent in self.req_model.get("entities", []):
            if isinstance(ent, dict) and ent.get("type", "").lower() in comp_types:
                discovered_entities.append((ent.get("name", ""), ent.get("type", "component"), ent.get("description", "")))
            elif hasattr(ent, "type") and getattr(ent, "type", "").lower() in comp_types:
                discovered_entities.append((getattr(ent, "name", ""), getattr(ent, "type", "component"), getattr(ent, "description", "")))

        for d_name, d_type, d_desc in discovered_entities:
            if d_name and not any(c.name.lower() == d_name.lower() for c in self.ir.components):
                comp_id = new_efs_id("comp")
                comp = EFSComponent(
                    component_id=comp_id,
                    name=d_name,
                    type=d_type.lower(),
                    description=d_desc or f"Architectural component: {d_name}",
                    traceability=self._find_entity_traceability(d_name)
                )
                self.ir.components.append(comp)
                self.name_to_component_id[d_name] = comp_id

        # 3. Add Top-Level Composite Design Component if not already present
        top_name = getattr(self.ir.metadata, "display_name", "") or getattr(self.ir.metadata, "design_name", "top_module")
        if not any(c.name.lower() == top_name.lower() for c in self.ir.components):
            top_comp_id = new_efs_id("comp")
            top_comp = EFSComponent(
                component_id=top_comp_id,
                name=top_name,
                type="top_level_design",
                description=f"Top-Level Composite Design Module: {top_name}",
                traceability=self._find_entity_traceability(top_name)
            )
            self.ir.components.insert(0, top_comp)
            self.name_to_component_id[top_name] = top_comp_id

    def _build_interfaces_and_signals(self):
        """Build port signals and group interfaces."""
        # Determine clock and reset signal configurations
        clk_sigs = []
        rst_sigs = []
        for cr in self.req_model.get("clocks_resets", []):
            sig_id = new_efs_id("sig")
            name = cr.get("name", "clk")
            sig = EFSSignal(
                signal_id=sig_id,
                name=name,
                width="1",
                direction="input",
                datatype="logic",
                semantic_role="clock" if cr.get("type") == "clock" else "reset",
                description=cr.get("description", ""),
                traceability=self._find_entity_traceability(name)
            )
            self.ir.signals.append(sig)
            self.name_to_signal_id[name] = sig_id
            if cr.get("type") == "clock":
                clk_sigs.append(name)
                self.ir.clocks.append(sig)
            else:
                rst_sigs.append(name)
                self.ir.resets.append(sig)

        primary_clk = clk_sigs[0] if clk_sigs else "clk"
        primary_rst = rst_sigs[0] if rst_sigs else "rst_n"

        # Map external interface port list from requirement model or design spec text
        ports = self.req_model.get("interfaces", [])
        if not ports and self.design_spec:
            try:
                from core.agents.rtl_agent import parse_markdown_spec
                parsed_spec = parse_markdown_spec(self.design_spec)
                ports = parsed_spec.get("interfaces", [])
            except Exception:
                pass

        interface_signals_ids = []
        for port in ports:
            name = str(port.get("name", "")).strip("` \t\r\n'\"")
            if name.lower() in ("name", "signal", "port", "signal name", "port name", "signal_name", "port_name", "pin", "width", "direction", "description", "protocol/description"):
                continue
            from core.verification.compatibility_engine import ProtocolRoleMapper

            sig_id = new_efs_id("sig")
            raw_dir = str(port.get("direction", "")).strip().lower()
            
            # Infer interface role (slave vs master)
            iface_role = "slave" if "slave" in self.design_spec.lower() or "target" in self.design_spec.lower() or "slave" in str(self.req_model).lower() else "master"
            role_info = ProtocolRoleMapper.get_semantic_role(name, iface_role)

            if "in" in raw_dir and "out" in raw_dir:
                direction = "inout"
            elif "out" in raw_dir:
                direction = "output"
            elif "in" in raw_dir:
                direction = "input"
            elif role_info:
                # Deduce correct direction from protocol role mapping if raw direction is absent/ambiguous
                direction = role_info[2]
            else:
                direction = "input"

            width = str(port.get("width", "1"))
            semantic_role = role_info[1] if role_info else ("control" if any(k in name.lower() for k in ("valid", "ready", "enable", "start", "done", "busy")) else "data")
            
            sig = EFSSignal(
                signal_id=sig_id,
                name=name,
                width=width,
                direction=direction,
                datatype="logic",
                clock=primary_clk,
                reset=primary_rst,
                description=port.get("description", ""),
                semantic_role=semantic_role,
                traceability=self._find_entity_traceability(name)
            )
            self.ir.signals.append(sig)
            self.name_to_signal_id[name] = sig_id
            interface_signals_ids.append(sig_id)

        # Create a generic external interface connection block based strictly on extracted signals
        if interface_signals_ids:
            iface_id = new_efs_id("if")
            iface = EFSInterface(
                interface_id=iface_id,
                name="external_bus_interface",
                protocol=self.ir.metadata.protocol,
                role="slave" if "slave" in self.design_spec.lower() or "target" in self.design_spec.lower() else "master",
                signals=interface_signals_ids,
                clock=primary_clk,
                reset=primary_rst,
                description="Primary bus port boundary interface.",
                traceability=self._find_entity_traceability(self.ir.metadata.protocol)
            )
            self.ir.interfaces.append(iface)
            self.name_to_interface_id[iface.name] = iface_id
            
            # Map interface back to components
            for comp in self.ir.components:
                if "controller" in comp.name.lower() or "fsm" in comp.name.lower():
                    comp.interfaces.append(iface_id)

        # Synchronize clocks and resets collections with canonical signals
        if not self.ir.clocks:
            self.ir.clocks = [s for s in self.ir.signals if s.semantic_role == "clock"]
        if not self.ir.resets:
            self.ir.resets = [s for s in self.ir.signals if s.semantic_role == "reset"]

    def _build_registers(self):
        """Map requirement model register layout."""
        for reg in self.req_model.get("registers", []):
            name = reg.get("name", "")
            if not is_valid_register_candidate(name, reg):
                continue
            reg_id = new_efs_id("reg")
            
            reg_width = reg.get("width", 32)
            if isinstance(reg_width, str):
                try:
                    reg_width = int(reg_width)
                except ValueError:
                    reg_width = 32

            fields = []
            used_bits = set()
            next_available_bit = 0

            for f in reg.get("fields", []):
                msb, lsb = None, None
                if "msb" in f and "lsb" in f:
                    try:
                        msb = int(f["msb"])
                        lsb = int(f["lsb"])
                    except (TypeError, ValueError):
                        pass

                if msb is None:
                    bits_str = str(f.get("bits", "")).strip()
                    if ":" in bits_str:
                        try:
                            parts = bits_str.replace("[", "").replace("]", "").split(":")
                            msb, lsb = int(parts[0]), int(parts[1])
                        except ValueError:
                            pass
                    elif bits_str.isdigit():
                        try:
                            msb = lsb = int(bits_str)
                        except ValueError:
                            pass

                f_width = f.get("width", 1)
                if isinstance(f_width, str) and f_width.isdigit():
                    f_width = int(f_width)
                if not isinstance(f_width, int) or f_width < 1:
                    f_width = abs((msb or 0) - (lsb or 0)) + 1 if (msb is not None and lsb is not None) else 1

                # If msb/lsb missing or overlaps already allocated bits, resolve to next available bit offset
                if msb is None or lsb is None:
                    while any((next_available_bit + i) in used_bits for i in range(f_width)):
                        next_available_bit += 1
                    lsb = next_available_bit
                    msb = lsb + f_width - 1
                else:
                    low, high = min(lsb, msb), max(lsb, msb)
                    if any(b in used_bits for b in range(low, high + 1)):
                        while any((next_available_bit + i) in used_bits for i in range(f_width)):
                            next_available_bit += 1
                        lsb = next_available_bit
                        msb = lsb + f_width - 1

                low, high = min(lsb, msb), max(lsb, msb)
                for b in range(low, high + 1):
                    used_bits.add(b)
                next_available_bit = max(next_available_bit, high + 1)

                fields.append(EFSRegisterField(
                    name=f.get("name", "FIELD"),
                    msb=high,
                    lsb=low,
                    width=(high - low + 1),
                    access=f.get("access", reg.get("access", "RW")),
                    reset_value=str(f.get("reset_val", f.get("reset_value", "0"))),
                    description=f.get("description", "")
                ))

            max_field_msb = max([f.msb for f in fields], default=31)
            if max_field_msb >= reg_width:
                reg_width = max_field_msb + 1

            norm_offset = normalize_hardware_offset(reg.get("offset"))
            efs_reg = EFSRegister(
                register_id=reg_id,
                name=name,
                offset=norm_offset,
                offset_source="explicit" if norm_offset is not None else "missing",
                width=reg_width,
                access_type=reg.get("access", "RW"),
                fields=fields,
                description=reg.get("description", ""),
                traceability=self._find_entity_traceability(name)
            )
            self.ir.registers.append(efs_reg)
            self.name_to_register_id[name] = reg_id

        # If no explicit registers were parsed from the document, keep registers as empty array without synthetic defaults

    def _build_fsms(self):
        """Build Finite State Machine structures."""
        fsm_info = self.req_model.get("fsm_info", {})
        if not fsm_info:
            return

        states = []
        for state_name in fsm_info.get("states", []):
            state_id = new_efs_id("st")
            
            # Find actions or outputs related to this state from the plan
            plan_actions = ""
            for s in self.plan.get("fsm_architecture", {}).get("states", []):
                if isinstance(s, dict) and s.get("name") == state_name:
                    plan_actions = s.get("actions", "")
            
            states.append(EFSState(
                state_id=state_id,
                name=state_name,
                actions=[plan_actions] if plan_actions else []
            ))

        transitions = []
        for trans in fsm_info.get("transitions", []):
            trans_id = new_efs_id("trans")
            transitions.append(EFSTransition(
                transition_id=trans_id,
                source_state=trans.get("from"),
                target_state=trans.get("to"),
                condition=trans.get("condition", "1"),
                traceability=self._find_relationship_traceability(trans.get("from"), trans.get("to"))
            ))

        fsm_id = new_efs_id("fsm")
        fsm = EFSFSM(
            fsm_id=fsm_id,
            name="Main_Controller_FSM",
            states=states,
            initial_state=states[0].name if states else "IDLE",
            transitions=transitions,
            traceability=self._find_entity_traceability("state machine")
        )
        self.ir.fsms.append(fsm)

    def _build_protocol_rules_and_constraints(self):
        """Map retrieved protocol spec chunks and relationships to ProtocolRules and Constraints."""
        # 1. Map retrieved protocol rules from database
        for r in self.protocol_rules:
            rule_id = new_efs_id("prule")
            self.ir.protocol_rules.append(EFSProtocolRule(
                rule_id=rule_id,
                name=r.get("citation", "Protocol Clause"),
                type="handshake" if "handshake" in r.get("text", "").lower() else "general",
                description=r.get("text", ""),
                traceability=EFSTraceability(
                    chunk_id=r.get("chunk_id"),
                    original_text=r.get("text"),
                    extraction_method="graphrag_retrieval",
                    confidence=r.get("score", 1.0)
                )
            ))

        # 2. Extract constraints from protocol rules or specifications
        # For example, AXI4 handshake stability or VALID/READY dependencies
        for p_sig in self.ir.signals:
            if "valid" in p_sig.name.lower():
                # Add handshake stability constraint
                const_id = new_efs_id("const")
                self.ir.constraints.append(EFSConstraint(
                    constraint_id=const_id,
                    type="stability",
                    source_objects=[p_sig.name],
                    condition=f"asserted({p_sig.name})",
                    expected_behavior=f"stable({p_sig.name}) until handshaked",
                    severity="CRITICAL",
                    protocol=self.ir.metadata.protocol,
                    traceability=self._find_entity_traceability(p_sig.name)
                ))

        # 3. Add timing constraints from requirement model
        for idx, tc in enumerate(self.req_model.get("timing_constraints", [])):
            rule_id = new_efs_id("time")
            self.ir.timing_rules.append(EFSTimingRule(
                rule_id=rule_id,
                type="cycle_relationship" if "cycle" in tc.lower() else "latency",
                cycle_relationship=tc,
                traceability=self._find_entity_traceability("timing")
            ))

    def _build_flows(self):
        """Construct transactional Flows representing sequence steps."""
        fsm_info = self.req_model.get("fsm_info", {})
        if not fsm_info or not fsm_info.get("states"):
            return

        steps = []
        for idx, state in enumerate(fsm_info.get("states", [])):
            step_id = new_efs_id("fstep")
            steps.append(EFSFlowStep(
                step_id=step_id,
                action_description=f"Controller transitions to state {state} to run operations.",
                state_transitions=[state]
            ))

        flow_id = new_efs_id("flow")
        flow = EFSFlow(
            flow_id=flow_id,
            name="Primary_Operation_Flow",
            ordered_steps=steps,
            traceability=self._find_entity_traceability("flow")
        )
        self.ir.flows.append(flow)

    def _build_transactions(self):
        """Extract and build formal hardware transaction definitions."""
        raw_txs = self.req_model.get("transactions", [])
        if not raw_txs:
            raw_txs = self.plan.get("transactions", [])

        # Also check entities in registry
        if not raw_txs and self.registry:
            for ent in self.registry.all_entities():
                if getattr(ent, "entity_type", "") in ("Transaction", "TransactionStep"):
                    raw_txs.append({
                        "name": ent.name,
                        "description": getattr(ent, "original_text", ""),
                        "type": "transaction"
                    })

        for tx_data in raw_txs:
            if isinstance(tx_data, dict):
                tx_name = tx_data.get("name", "unnamed_transaction")
                tx_id = new_efs_id("txn")
                resp_val = tx_data.get("expected_response", "") or tx_data.get("response", "")
                trace = self._find_entity_traceability(tx_name)
                self.ir.transactions.append(EFSTransaction(
                    transaction_id=tx_id,
                    name=tx_name,
                    initiator=tx_data.get("initiator", ""),
                    target=tx_data.get("target", ""),
                    signals=tx_data.get("signals", []),
                    response=resp_val,
                    expected_response=resp_val,
                    description=tx_data.get("description", ""),
                    traceability=trace
                ))

    def _build_assertion_intent(self):
        """Compile SVA intent assertions from constraints without hardcoded protocol fallbacks."""
        primary_clk = next((s.name for s in self.ir.signals if s.semantic_role == "clock"), "clk")
        primary_rst = next((s.name for s in self.ir.signals if s.semantic_role == "reset"), "rst_n")

        for const in self.ir.constraints:
            if const.type in ("stability", "handshake"):
                ast_id = new_efs_id("ast")
                valid_sigs = [s.name for s in self.ir.signals if "valid" in s.name.lower()]
                rdy_sigs = [s.name for s in self.ir.signals if "ready" in s.name.lower()]

                if const.source_objects:
                    sig_name = const.source_objects[0]
                elif valid_sigs:
                    sig_name = valid_sigs[0]
                else:
                    continue  # Skip handshake assertion if no signal is specified or present

                matched_rdy = rdy_sigs[0] if rdy_sigs else ""
                antecedent_str = f"{sig_name} && !{matched_rdy}" if matched_rdy else f"asserted({sig_name})"

                self.ir.assertions.append(EFSAssertionIntent(
                    assertion_id=ast_id,
                    derived_constraint_id=const.constraint_id,
                    property_type="assert",
                    clock=primary_clk,
                    reset_condition=primary_rst,
                    antecedent=antecedent_str,
                    consequent=f"$stable({sig_name})",
                    temporal_behavior="implication",
                    severity="error",
                    traceability=const.traceability
                ))

    def _build_coverage_intent(self):
        """Map coverage items to FSM states and transitions."""
        for fsm in self.ir.fsms:
            # 1. State Coverage
            cov_id = new_efs_id("cov")
            self.ir.coverage.append(EFSCoverageIntent(
                coverage_id=cov_id,
                type="state",
                target_object_id=fsm.fsm_id,
                description=f"Verify coverage of all states in FSM '{fsm.name}': {', '.join([s.name for s in fsm.states])}.",
                traceability=fsm.traceability
            ))

            # 2. Transition Coverage
            cov_id = new_efs_id("cov")
            self.ir.coverage.append(EFSCoverageIntent(
                coverage_id=cov_id,
                type="transition",
                target_object_id=fsm.fsm_id,
                description=f"Verify coverage of all state-to-state transitions in FSM '{fsm.name}'.",
                traceability=fsm.traceability
            ))

    def _build_instructions(self):
        """Map instructions from req_model."""
        for instr in self.req_model.get("instructions", []):
            instr_id = new_efs_id("instr")
            name = (instr.get("name") or instr.get("mnemonic") or "").strip()
            if not name:
                continue
            fields = []
            for f in instr.get("fields", []):
                fields.append(EFSInstructionField(
                    name=f.get("name", "FIELD"),
                    width=f.get("width", 0),
                    msb=f.get("msb", 0),
                    lsb=f.get("lsb", 0),
                    description=f.get("description", "")
                ))
            raw_opcode = (
                instr.get("opcode")
                or instr.get("attributes", {}).get("opcode")
                or instr.get("attributes", {}).get("opcode_hex")
                or instr.get("attributes", {}).get("binary_encoding")
                or ""
            )
            opcode_str = str(raw_opcode).strip() if raw_opcode is not None else ""
            if opcode_str.lower() in ("unknown", "none", "n/a", "unresolved", "tbd", "missing", "nan", "null", "-"):
                opcode_str = ""

            self.ir.instructions.append(EFSInstruction(
                instruction_id=instr_id,
                name=name,
                mnemonic=instr.get("mnemonic") or name,
                opcode=opcode_str,
                width=instr.get("width", 32),
                fields=fields,
                description=instr.get("description") or instr.get("attributes", {}).get("description", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_opcodes(self):
        """Map opcodes from req_model with strict property resolution and cross-linking."""
        for opc in self.req_model.get("opcodes", []):
            opc_id = new_efs_id("opc")
            mnemonic = (
                opc.get("mnemonic")
                or opc.get("attributes", {}).get("instruction")
                or opc.get("attributes", {}).get("mnemonic")
                or opc.get("name", "")
                or ""
            ).strip()
            # Clean generic '_OPCODE' suffix if entity was created as f"{NAME}_OPCODE"
            if mnemonic.upper().endswith("_OPCODE"):
                mnemonic = mnemonic[:-7].strip()
            elif mnemonic.lower().endswith("_opcode"):
                mnemonic = mnemonic[:-7].strip()

            if not mnemonic:
                continue

            raw_encoding = (
                opc.get("binary_encoding")
                or opc.get("attributes", {}).get("opcode_hex")
                or opc.get("attributes", {}).get("binary_encoding")
                or opc.get("attributes", {}).get("encoding")
                or opc.get("attributes", {}).get("value")
                or opc.get("opcode")
                or ""
            )
            encoding_str = str(raw_encoding).strip() if raw_encoding is not None else ""
            if encoding_str.lower() in ("unknown", "none", "n/a", "unresolved", "tbd", "missing", "nan", "null", "-"):
                encoding_str = ""

            # Check if matching instruction exists to cross-link
            matching_instr = next(
                (i for i in self.ir.instructions if i.name.upper() == mnemonic.upper() or i.mnemonic.upper() == mnemonic.upper()),
                None
            )
            instr_id = matching_instr.instruction_id if matching_instr else None

            # Cross-resolve opcode if one representation had it and the other was empty
            if matching_instr and matching_instr.opcode and not encoding_str:
                encoding_str = matching_instr.opcode
            elif matching_instr and encoding_str and not matching_instr.opcode:
                matching_instr.opcode = encoding_str

            trace = self._find_entity_traceability(mnemonic)
            efs_opc = EFSOpcode(
                opcode_id=opc_id,
                mnemonic=mnemonic,
                binary_encoding=encoding_str,
                operation=opc.get("operation") or opc.get("attributes", {}).get("operation", ""),
                operands=opc.get("operands") or opc.get("attributes", {}).get("operands", []),
                instruction_id=instr_id,
                traceability=trace
            )
            self.ir.opcodes.append(efs_opc)
            logger.debug(
                f"[OPCODE_BUILD] mnemonic='{mnemonic}' | encoding='{encoding_str}' | "
                f"opc_id='{opc_id}' | instr_id='{instr_id}' | source_chunk='{getattr(trace, 'chunk_id', '')}'"
            )

    def _build_memory_regions(self):
        """Map memory regions from req_model."""
        for mr in self.req_model.get("memory_regions", []):
            mr_id = new_efs_id("memreg")
            name = mr.get("name", "memory_region")
            self.ir.memory_regions.append(EFSMemoryRegion(
                region_id=mr_id,
                name=name,
                type=mr.get("type", "SRAM"),
                base_address=mr.get("base_address", ""),
                size=mr.get("size", ""),
                depth=mr.get("depth", 0),
                width=mr.get("width", 0),
                description=mr.get("description", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_data_structures(self):
        """Map data structures from req_model."""
        for ds in self.req_model.get("data_structures", []):
            ds_id = new_efs_id("dstruct")
            name = ds.get("name", "data_structure")
            self.ir.data_structures.append(EFSDataStructure(
                structure_id=ds_id,
                name=name,
                fields=ds.get("fields", []),
                description=ds.get("description", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_errors(self):
        """Map errors from req_model."""
        for err in self.req_model.get("errors", []):
            err_id = new_efs_id("err")
            name = err.get("name", "error")
            self.ir.errors.append(EFSError(
                error_id=err_id,
                name=name,
                code=err.get("code", ""),
                condition=err.get("condition", ""),
                behavior=err.get("behavior", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_performance_requirements(self):
        """Map performance requirements from req_model."""
        for pr in self.req_model.get("performance_requirements", []):
            pr_id = new_efs_id("perf")
            metric = pr.get("metric", "throughput")
            self.ir.performance_requirements.append(EFSPerformanceRequirement(
                perf_id=pr_id,
                metric=metric,
                target_value=pr.get("target_value", ""),
                condition=pr.get("condition", ""),
                traceability=self._find_entity_traceability(metric)
            ))

    def _build_conflicts(self):
        """Map extracted conflicts and perform programmatic checks."""
        run_id = getattr(self.ir.metadata, "generation_run_id", "")

        # 1. Map conflicts from req_model
        for conf in self.req_model.get("spec_conflicts", []):
            conflict_id = new_efs_id("conf")
            self.ir.conflicts.append(EFSConflict(
                conflict_id=conflict_id,
                type=conf.get("type", "AMBIGUITY"),
                severity=conf.get("severity", "MAJOR"),
                generation_run_id=run_id,
                source_objects=conf.get("source_objects", []),
                evidence=conf.get("evidence", ""),
                description=conf.get("description", ""),
                candidates=conf.get("candidates", []),
                conflicting_values=conf.get("conflicting_values", []),
                resolution_status="UNRESOLVED",
                traceability=self._find_entity_traceability(conf.get("type", "conflict"))
            ))

        # 2. Programmatic Check: Duplicate Opcode Assignments
        opcode_map = {}
        for opc in self.ir.opcodes:
            enc = opc.binary_encoding.strip()
            if enc and enc != "UNKNOWN":
                opcode_map.setdefault(enc, []).append(opc)
                
        for enc, opcs in opcode_map.items():
            if len(opcs) > 1:
                conflict_id = new_efs_id("conf")
                mnemonics = [o.mnemonic for o in opcs]
                ids = [o.opcode_id for o in opcs]
                self.ir.conflicts.append(EFSConflict(
                    conflict_id=conflict_id,
                    type="DUPLICATE_OPCODE",
                    severity="CRITICAL",
                    blocking=True,
                    generation_run_id=run_id,
                    source_objects=ids,
                    evidence=f"Opcode binary value '{enc}' is assigned to multiple instructions: {', '.join(mnemonics)}",
                    description=f"Duplicate opcode assignment detected for value {enc}",
                    candidates=mnemonics,
                    conflicting_values=[enc],
                    resolution_status="UNRESOLVED",
                    traceability=EFSTraceability(extraction_method="programmatic_check", confidence=1.0)
                ))

        # 3. Programmatic Check: Overlapping Register Fields
        for reg in self.ir.registers:
            field_bits = {}
            for f in reg.fields:
                for bit in range(f.lsb, f.msb + 1):
                    field_bits.setdefault(bit, []).append(f.name)
            overlaps = {bit: names for bit, names in field_bits.items() if len(names) > 1}
            if overlaps:
                conflict_id = new_efs_id("conf")
                overlap_desc = "; ".join([f"bit {b} in {', '.join(names)}" for b, names in overlaps.items()])
                self.ir.conflicts.append(EFSConflict(
                    conflict_id=conflict_id,
                    type="OVERLAPPING_REGISTER_FIELDS",
                    severity="CRITICAL",
                    blocking=True,
                    generation_run_id=run_id,
                    source_objects=[reg.register_id],
                    evidence=f"Register '{reg.name}' has overlapping field definitions: {overlap_desc}",
                    description=f"Register '{reg.name}' has fields sharing the same bit locations",
                    resolution_status="UNRESOLVED",
                    traceability=EFSTraceability(extraction_method="programmatic_check", confidence=1.0)
                ))

        # 4. Programmatic Check: Duplicate Register Addresses (Faithful Preservation)
        def _normalize_offset_key(raw: Any) -> str:
            if raw is None:
                return ""
            s = str(raw).strip()
            if not s:
                return ""
            try:
                if s.lower().startswith("0x"):
                    return f"0x{int(s, 16):x}"
                elif s.isdigit():
                    return f"0x{int(s, 10):x}"
            except ValueError:
                pass
            return s.lower()

        reg_offsets: Dict[str, List[Any]] = {}
        for reg in self.ir.registers:
            if not is_explicit_address(reg.offset):
                continue
            off_key = _normalize_offset_key(reg.offset)
            if off_key:
                reg_offsets.setdefault(off_key, []).append(reg)

        for off_val, regs in reg_offsets.items():
            if len(regs) > 1:
                # Check for intentional alias, mirror, shadow, banked, or view semantics
                is_intentional_alias = any(
                    any(term in (r.description or "").lower() for term in ("alias", "shadow", "mirror", "banked", "alternate view", "register view"))
                    for r in regs
                )
                if is_intentional_alias:
                    continue
                names = [r.name for r in regs]
                conf_id = new_efs_id("conf")
                candidates = []
                for r in regs:
                    other_names = [o.name for o in regs if o.name != r.name]
                    candidates.append(EFSResolutionCandidate(
                        action="REASSIGN_VALUE",
                        target_entity=r.name,
                        property="offset",
                        current_value=r.offset,
                        proposed_value=None,
                        reason=f"Reassign unique offset to register '{r.name}' to resolve collision with {', '.join(other_names)} at {off_val}.",
                        confidence=None,
                        validation_status="PENDING"
                    ))
                self.ir.conflicts.append(EFSConflict(
                    conflict_id=conf_id,
                    type="DUPLICATE_REGISTER_ADDRESS",
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    resolution_status="UNRESOLVED",
                    requires_human_decision=True,
                    generation_run_id=run_id,
                    title=f"Duplicate Register Address: {', '.join(names)}",
                    entities=names,
                    affected_entities=names,
                    target_entities=names,
                    property="offset",
                    properties=["offset"],
                    conflicting_values=[r.offset for r in regs],
                    current_values={r.name: r.offset for r in regs},
                    source_objects=[r.register_id for r in regs],
                    resolution_candidates=candidates,
                    ai_proposal={
                        "action": "HUMAN_RESOLUTION_REQUIRED",
                        "target_entity": names[1] if len(names) > 1 else names[0],
                        "property": "offset",
                        "current_value": regs[1].offset if len(regs) > 1 else regs[0].offset,
                        "proposed_value": None,
                        "reason": f"Address collision detected between {', '.join(names)} at offset '{off_val}'. Human architect must assign non-overlapping offset."
                    },
                    source_references=[
                        {
                            "entity": r.name,
                            "source": getattr(r.traceability, "source_document", "Specification") or "Specification",
                            "section": "Register Map",
                            "value": r.offset
                        }
                        for r in regs
                    ],
                    evidence=f"Registers {', '.join(names)} share the same address offset '{off_val}'.",
                    description=f"Address map conflict: Multiple registers ({', '.join(names)}) assigned to identical offset '{off_val}'.",
                    explanation=f"Address collision detected between {', '.join(names)}. Writing to offset '{off_val}' would mutate multiple registers simultaneously.",
                    impact="Address bus decoding collision resulting in unintended register writes and corrupted states.",
                    suggestions=[f"Reassign unique address offset for {r.name}" for r in regs],
                    traceability=regs[0].traceability
                ))

    def _build_extraction_quality(self):
        """Assess extraction quality honestly without falsely claiming completeness."""
        missing = []
        if not self.ir.signals:
            missing.append("signals")
        if not self.ir.components:
            missing.append("components")
        
        # Check if specification explicitly expects registers / instructions / FSMs
        spec_lower = self.design_spec.lower()
        if ("register" in spec_lower or "csr" in spec_lower) and not self.ir.registers:
            missing.append("registers")
        if ("instruction" in spec_lower or "opcode" in spec_lower) and not (self.ir.instructions or self.ir.opcodes):
            missing.append("instructions")
        if ("state machine" in spec_lower or " fsm " in spec_lower) and not self.ir.fsms:
            missing.append("fsms")
            
        has_primary = bool(self.ir.signals and self.ir.components)
        has_behavioral = bool(self.ir.registers or self.ir.fsms or self.ir.instructions or self.ir.transactions)
        has_blocking_conflicts = any(
            c.blocking and c.resolution_status != "RESOLVED"
            for c in self.ir.conflicts
        )

        if not has_primary:
            extraction_status = "failed" if not self.ir.signals and not self.ir.components else "partial"
        elif has_blocking_conflicts:
            extraction_status = "partial"
        else:
            extraction_status = "success"

        if has_blocking_conflicts or missing:
            ir_comp = "partial"
        elif has_primary and has_behavioral:
            ir_comp = "complete"
        else:
            ir_comp = "partial"

        if not has_primary or has_blocking_conflicts:
            src_comp = "partial" if has_primary else "unknown"
        elif not missing and has_behavioral:
            src_comp = "complete"
        else:
            src_comp = "partial"
        
        self.ir.extraction_quality = EFSExtractionQuality(
            extraction_status=extraction_status,
            source_completeness=src_comp,
            ir_completeness=ir_comp,
            missing_information=missing
        )
