"""
Generic Semantic Categories and Taxonomy Registry.

Defines protocol-independent semantic categories and allows dynamic
extension during specification ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


# Standard generic semantic categories required by the target architecture
DEFAULT_SEMANTIC_CATEGORIES = [
    "ENTITY", "COMPONENT", "SUBCOMPONENT", "SIGNAL", "PORT", "REGISTER", "REGISTER_FIELD", "PARAMETER",
    "INTERFACE", "TRANSACTION", "OPERATION", "COMMAND", "EVENT", "STATE",
    "TRANSITION", "CONDITION", "GUARD", "ACTION", "TIMING", "CLOCK", "RESET",
    "HANDSHAKE", "PROTOCOL_RULE", "DATA_FORMAT", "MEMORY", "RESOURCE",
    "CONSTRAINT", "ERROR", "EXCEPTION", "INTERRUPT", "SEQUENCE", "LOOP",
    "WAIT", "TIMEOUT", "SUBFLOW", "DEPENDENCY", "CONFIGURATION", "PERFORMANCE",
    "POWER", "SECURITY", "INSTRUCTION", "INSTRUCTION_FORMAT", "INSTRUCTION_FIELD", "OPCODE",
    "MEMORY_RESOURCE", "MEMORY_REGION", "CACHE", "BUFFER", "ADDRESS_SPACE",
    "DATA_STRUCTURE", "DATA_FIELD", "EXECUTION_STRATEGY", "SCHEDULING_POLICY", "SYNCHRONIZATION",
    "THROUGHPUT", "LATENCY", "CLOCK_FREQUENCY", "POWER_REQUIREMENT", "BANDWIDTH", "CAPACITY",
    "FUNCTIONAL_REQUIREMENT", "ARCHITECTURAL_REQUIREMENT", "INTERFACE_REQUIREMENT",
    "RESOURCE_REQUIREMENT", "CONCURRENCY_REQUIREMENT",
    "CUSTOM_ENTITY", "UNKNOWN", "UNCLASSIFIED", "CUSTOM"
]

DEFAULT_ENTITY_TYPES = [
    "component", "subcomponent", "signal", "register", "field", "module", "interface", "transaction",
    "operation", "state", "event", "parameter", "resource", "memory",
    "clock", "reset", "command", "message", "packet", "channel",
    "subsystem", "fsm", "counter", "arbiter", "fifo", "buffer",
    "instruction", "instruction_format", "instruction_field", "opcode",
    "memory_resource", "memory_region", "cache", "address_space",
    "data_structure", "data_field", "execution_strategy", "scheduling_policy", "synchronization",
    "error", "exception",
    "performance_requirement", "throughput", "latency", "clock_frequency", "power_requirement", "bandwidth", "capacity",
    "functional_requirement", "architectural_requirement", "interface_requirement", "resource_requirement",
    "scheduler", "engine", "queue", "custom_entity"
]

DEFAULT_RELATIONSHIP_TYPES = [
    "DEPENDS_ON", "REFERENCES", "USES", "PRODUCES", "CONSUMES", "TRIGGERS",
    "FOLLOWED_BY", "PRECEDES", "CAUSES", "TRANSITIONS_TO", "CONSTRAINS",
    "CONFIGURES", "CONTROLS", "BELONGS_TO", "PART_OF", "ASSOCIATED_WITH",
    "USES_SIGNAL", "USES_REGISTER", "USES_EVENT", "USES_STATE"
]


@dataclass
class SemanticTypeSpec:
    name: str
    attributes: List[str] = field(default_factory=list)
    description: str = ""


class TaxonomyRegistry:
    """Registry managing semantic categories, entity types, and dynamic categories."""

    def __init__(self):
        self._categories: Set[str] = set(DEFAULT_SEMANTIC_CATEGORIES)
        self._entity_types: Set[str] = set(DEFAULT_ENTITY_TYPES)
        self._relationship_types: Set[str] = set(DEFAULT_RELATIONSHIP_TYPES)
        self._custom_specs: Dict[str, SemanticTypeSpec] = {}

    def is_valid_category(self, cat: str) -> bool:
        return cat.upper() in self._categories or cat.upper() in self._custom_specs

    def register_category(self, name: str, attributes: List[str] = None, description: str = "") -> None:
        upper_name = name.upper()
        self._categories.add(upper_name)
        self._custom_specs[upper_name] = SemanticTypeSpec(
            name=upper_name,
            attributes=attributes or [],
            description=description
        )

    def register_entity_type(self, entity_type: str) -> None:
        self._entity_types.add(entity_type.lower())

    def register_relationship_type(self, rel_type: str) -> None:
        self._relationship_types.add(rel_type.upper())

    def get_all_categories(self) -> List[str]:
        return sorted(list(self._categories))

    def get_all_entity_types(self) -> List[str]:
        return sorted(list(self._entity_types))

    def get_all_relationship_types(self) -> List[str]:
        return sorted(list(self._relationship_types))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categories": self.get_all_categories(),
            "entity_types": self.get_all_entity_types(),
            "relationship_types": self.get_all_relationship_types(),
            "custom_specs": {k: {"name": v.name, "attributes": v.attributes, "description": v.description}
                             for k, v in self._custom_specs.items()}
        }


# Singleton instance
TAXONOMY = TaxonomyRegistry()
