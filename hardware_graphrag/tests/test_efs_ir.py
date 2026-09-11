"""
Unit and Integration Tests for EFS IR Layer.
"""

from __future__ import annotations

import json
import os
import unittest

from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSTraceability, EFSComponent, EFSInterface, EFSSignal,
    EFSRegister, EFSRegisterField, EFSFSM, EFSState, EFSTransition,
    EFSConstraint, EFSTimingRule, EFSProtocolRule, EFSFlow, EFSFlowStep,
    EFSAssertionIntent, EFSCoverageIntent
)
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.validator import validate_efs_ir


class TestEFSIR(unittest.TestCase):
    """Test suite checking creation, validation, and serialization of EFS IR."""

    def test_efs_ir_creation_and_serialization(self):
        """Create a complete EFS IR object, serialize to JSON, and deserialize back."""
        ir = EFSIR()
        ir.metadata = EFSMetadata(
            design_name="DMA_Controller",
            protocol="AXI4",
            protocol_version="1.0"
        )
        
        # Add component
        comp_id = "comp_dma_ctrl"
        comp = EFSComponent(
            component_id=comp_id,
            name="DMA_FSM_Controller",
            type="controller",
            description="Coordinates read/write handshakes",
            interfaces=["if_axi_slave"]
        )
        ir.components.append(comp)
        
        # Add interface
        iface = EFSInterface(
            interface_id="if_axi_slave",
            name="axi_slave_port",
            protocol="AXI4",
            role="slave",
            signals=["sig_awvalid", "sig_awready"]
        )
        ir.interfaces.append(iface)
        
        # Add signals
        sig1 = EFSSignal(
            signal_id="sig_awvalid",
            name="s_axi_awvalid",
            width="1",
            direction="input",
            semantic_role="control",
            interface="if_axi_slave",
            owner=comp_id
        )
        sig2 = EFSSignal(
            signal_id="sig_awready",
            name="s_axi_awready",
            width="1",
            direction="output",
            semantic_role="control",
            interface="if_axi_slave",
            owner=comp_id
        )
        ir.signals.extend([sig1, sig2])
        
        # Add register
        reg = EFSRegister(
            register_id="reg_control",
            name="CTRL_REG",
            offset="0x00",
            width=32,
            access_type="RW"
        )
        reg.fields.append(EFSRegisterField(
            name="EN",
            msb=0,
            lsb=0,
            width=1,
            access="RW",
            reset_value="0",
            description="Enable bit"
        ))
        ir.registers.append(reg)
        
        # Serialize to dict and verify
        serialized = ir.to_dict()
        self.assertEqual(serialized["metadata"]["design_name"], "DMA_Controller")
        self.assertEqual(len(serialized["components"]), 1)
        self.assertEqual(len(serialized["signals"]), 2)
        self.assertEqual(len(serialized["registers"][0]["fields"]), 1)
        
        # Deserialize
        deserialized = EFSIR.from_dict(serialized)
        self.assertEqual(deserialized.metadata.design_name, "DMA_Controller")
        self.assertEqual(deserialized.components[0].name, "DMA_FSM_Controller")
        self.assertEqual(deserialized.signals[0].name, "s_axi_awvalid")
        self.assertEqual(deserialized.registers[0].fields[0].name, "EN")

    def test_efs_ir_validation(self):
        """Check that the validator correctly flags issues like duplicate IDs and missing references."""
        ir = EFSIR()
        
        # 1. Flag missing metadata design_id
        ir.metadata.design_id = ""
        issues = validate_efs_ir(ir)
        self.assertTrue(any(i["category"] == "Metadata" for i in issues))
        
        # Reset ID
        ir.metadata.design_id = "dsn_test"
        
        # 2. Flag duplicate ID
        comp1 = EFSComponent(component_id="dup_id", name="comp1")
        comp2 = EFSComponent(component_id="dup_id", name="comp2")
        ir.components.extend([comp1, comp2])
        
        issues = validate_efs_ir(ir)
        self.assertTrue(any("Duplicate object ID" in i["message"] for i in issues))
        
        # Reset components
        ir.components = [comp1]
        comp1.component_id = "comp1"
        
        # 3. Flag invalid interface reference on component
        comp1.interfaces = ["if_missing"]
        issues = validate_efs_ir(ir)
        self.assertTrue(any("references non-existent interface" in i["message"] for i in issues))
        
        # 4. Flag invalid signal direction
        sig = EFSSignal(signal_id="sig1", name="test_sig", direction="invalid_dir")
        ir.signals.append(sig)
        issues = validate_efs_ir(ir)
        self.assertTrue(any("invalid direction" in i["message"] for i in issues))

    def test_efs_ir_builder(self):
        """Build EFS IR from mock requirement and planning models and verify it contains mapped objects."""
        req_model = {
            "protocol_references": ["AXI4"],
            "clocks_resets": [
                {"name": "clk", "type": "clock", "active_level": "rising", "description": "Global clk"},
                {"name": "rst_n", "type": "reset", "active_level": "low", "description": "Active-low reset"}
            ],
            "interfaces": [
                {"name": "s_axi_awvalid", "width": "1", "direction": "input", "description": "Address write valid"},
                {"name": "s_axi_awaddr", "width": "32", "direction": "input", "description": "Address write byte address"}
            ],
            "registers": [
                {
                    "name": "STATUS_REG",
                    "offset": "0x04",
                    "access": "RO",
                    "description": "Status indications",
                    "fields": [
                        {"name": "BUSY", "bits": "0", "description": "Engine busy", "reset_val": "0"}
                    ]
                }
            ],
            "fsm_info": {
                "states": ["IDLE", "SETUP", "ACTIVE"],
                "transitions": [
                    {"from": "IDLE", "to": "SETUP", "condition": "start_pulse == 1"}
                ]
            }
        }
        
        plan = {
            "system_architecture": "DMA_Core",
            "submodules": [
                {"name": "Register_File_Slice", "purpose": "Control decoding"}
            ]
        }
        
        protocol_rules = [
            {"chunk_id": "chunk_rule_1", "text": "AXI Handshake requires VALID stability.", "citation": "AMBA spec section A3.2"}
        ]
        
        builder = EFSIRBuilder(req_model, plan, protocol_rules)
        ir = builder.build()
        
        # Assert metadata mapping
        self.assertEqual(ir.metadata.protocol, "AXI4")
        
        # Assert FSM state mapping
        self.assertEqual(len(ir.fsms), 1)
        self.assertEqual(ir.fsms[0].initial_state, "IDLE")
        self.assertEqual(len(ir.fsms[0].states), 3)
        self.assertEqual(len(ir.fsms[0].transitions), 1)
        self.assertEqual(ir.fsms[0].transitions[0].source_state, "IDLE")
        
        # Assert registers mapping
        self.assertEqual(len(ir.registers), 1)
        self.assertEqual(ir.registers[0].name, "STATUS_REG")
        self.assertEqual(ir.registers[0].fields[0].name, "BUSY")
        
        # Assert signal definitions
        self.assertTrue(any(s.name == "s_axi_awaddr" and s.width == "32" for s in ir.signals))

    def test_extended_canonical_models_and_conflicts(self):
        """Test extended canonical models (instructions, opcodes, memory regions, conflicts) and the scorecard validator."""
        from core.efs_ir.models import EFSInstruction, EFSInstructionField, EFSOpcode, EFSMemoryRegion, EFSConflict
        from core.efs_ir.validator import get_efs_ir_quality
        
        ir = EFSIR()
        ir.metadata.design_id = "DSN_EXT_TEST"
        
        # 1. Add instructions and fields
        instr = EFSInstruction(
            name="MSTORE",
            width=32,
            fields=[
                EFSInstructionField(name="OPCODE", width=8, msb=31, lsb=24),
                EFSInstructionField(name="ADDR", width=24, msb=23, lsb=0)
            ]
        )
        ir.instructions.append(instr)
        
        # 2. Add duplicate opcodes to test programmatic check
        opc1 = EFSOpcode(mnemonic="MSTORE", binary_encoding="8'b00000100")
        opc2 = EFSOpcode(mnemonic="MINV", binary_encoding="8'b00000100")
        ir.opcodes.extend([opc1, opc2])
        
        # 3. Add memory region
        mr = EFSMemoryRegion(name="SRAM_0", type="SRAM", base_address="0x4000", size="16KB")
        ir.memory_regions.append(mr)
        
        # Build using builder programmatic checks by invoking EFSIRBuilder
        req_model = {
            "instructions": [
                {
                    "name": "MSTORE",
                    "width": 32,
                    "fields": [
                        {"name": "OPCODE", "width": 8, "msb": 31, "lsb": 24},
                        {"name": "ADDR", "width": 24, "msb": 23, "lsb": 0}
                    ]
                }
            ],
            "opcodes": [
                {"mnemonic": "MSTORE", "binary_encoding": "8'b00000100"},
                {"mnemonic": "MINV", "binary_encoding": "8'b00000100"}
            ],
            "memory_regions": [
                {"name": "SRAM_0", "type": "SRAM", "base_address": "0x4000", "size": "16KB"}
            ]
        }
        builder = EFSIRBuilder(req_model, {}, [])
        built_ir = builder.build()
        
        # Verify instructions, opcodes and memory regions mapped
        self.assertEqual(len(built_ir.instructions), 1)
        self.assertEqual(built_ir.instructions[0].name, "MSTORE")
        self.assertEqual(len(built_ir.opcodes), 2)
        
        # Verify programmatic check detected duplicate opcode
        self.assertTrue(any(c.type == "DUPLICATE_OPCODE" for c in built_ir.conflicts))
        
        # Validate and check quality scorecard
        issues = validate_efs_ir(built_ir)
        # Should have a conflict reported as issue
        self.assertTrue(any(i["category"] == "Conflicts" for i in issues))
        
        scorecard = get_efs_ir_quality(built_ir, issues)
        self.assertEqual(scorecard["status"], "BLOCKED")
        self.assertTrue(scorecard["critical_conflicts"] > 0)


if __name__ == "__main__":
    unittest.main()
