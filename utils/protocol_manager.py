"""
Protocol manager for switching between different hardware protocol specifications.

Manages data/protocols/library.json and handles dynamic path adjustments for
CONFIG and database connections to isolate protocols.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

from config import CONFIG
from core.vectorstore.faiss_store import reset_vector_store
from core.graph.neo4j_builder import reset_graph_store
from core.registry import reset_registry, get_registry
from utils.logger import get_logger

logger = get_logger("utils.protocol_manager")

PROTOCOLS_DIR = os.path.join(CONFIG.data_dir, "protocols")
LIBRARY_JSON_PATH = os.path.join(PROTOCOLS_DIR, "library.json")

DEFAULT_PROTOCOLS = ["AXI4", "AXI5", "APB", "AHB", "PCIe", "USB", "SPI", "I2C"]


def get_library_data() -> Dict[str, Any]:
    """Read the protocol library configuration."""
    os.makedirs(PROTOCOLS_DIR, exist_ok=True)
    if not os.path.exists(LIBRARY_JSON_PATH):
        # Create default library structure
        lib = {}
        for p in DEFAULT_PROTOCOLS:
            lib[p] = {
                "name": p,
                "filename": None,
                "chunks": 0,
                "entities": 0,
                "relationships": 0,
                "created_at": None,
                "reuse_count": 0,
                "status": "Not Uploaded"
            }
        save_library_data(lib)
        return lib

    try:
        with open(LIBRARY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load library JSON: {e}")
        return {}


def save_library_data(lib: Dict[str, Any]) -> None:
    """Save the protocol library configuration."""
    os.makedirs(PROTOCOLS_DIR, exist_ok=True)
    try:
        with open(LIBRARY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(lib, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save library JSON: {e}")


def bootstrap_library() -> None:
    """Bootstrap the protocol library using the existing data in the root directories as AXI4."""
    lib = get_library_data()
    axi4_dir = os.path.join(PROTOCOLS_DIR, "AXI4")

    # If AXI4 is already populated/ready, do nothing
    if lib.get("AXI4", {}).get("status") == "Ready" and os.path.exists(axi4_dir):
        return

    # Check if we have existing root data to bootstrap from
    root_graph_cache = CONFIG.graph_cache_dir
    root_vector_index = CONFIG.vector_index_dir

    registry_path = os.path.join(root_graph_cache, "registry.pkl")
    inmemory_graph_path = os.path.join(root_graph_cache, "inmemory_graph.pkl")
    faiss_index_path = os.path.join(root_vector_index, "index.faiss")

    has_data = os.path.exists(registry_path) or os.path.exists(inmemory_graph_path) or os.path.exists(faiss_index_path)

    if has_data:
        logger.info("Found existing GraphRAG data at root. Bootstrapping AXI4 protocol library...")
        os.makedirs(os.path.join(axi4_dir, "graph_cache"), exist_ok=True)
        os.makedirs(os.path.join(axi4_dir, "vector_index"), exist_ok=True)

        # Copy graph cache files
        if os.path.exists(root_graph_cache):
            for fn in os.listdir(root_graph_cache):
                src = os.path.join(root_graph_cache, fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(axi4_dir, "graph_cache", fn))

        # Copy vector index files
        if os.path.exists(root_vector_index):
            for fn in os.listdir(root_vector_index):
                src = os.path.join(root_vector_index, fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(axi4_dir, "vector_index", fn))

        # Dynamically load registry to get counts
        # Switch temporarily to AXI4 directory to load it
        orig_graph_dir = CONFIG.graph_cache_dir
        orig_vector_dir = CONFIG.vector_index_dir
        
        CONFIG.graph_cache_dir = os.path.join(axi4_dir, "graph_cache")
        CONFIG.vector_index_dir = os.path.join(axi4_dir, "vector_index")
        
        reset_registry()
        reg = get_registry()
        
        chunks_cnt = len(reg.all_chunks())
        entities_cnt = len({e.name for e in reg.all_entities()})
        rels_cnt = len(reg.all_relationships())
        
        filenames = [doc.filename for doc in reg.documents.values()]
        filename = filenames[0] if filenames else "protocol_specification.docx"

        # Restore original paths
        CONFIG.graph_cache_dir = orig_graph_dir
        CONFIG.vector_index_dir = orig_vector_dir
        reset_registry()

        lib["AXI4"] = {
            "name": "AXI4",
            "filename": filename,
            "chunks": chunks_cnt,
            "entities": entities_cnt,
            "relationships": rels_cnt,
            "created_at": datetime.now().isoformat(),
            "reuse_count": 1,
            "status": "Ready"
        }
        save_library_data(lib)
        logger.info(f"AXI4 protocol library bootstrapped with {chunks_cnt} chunks, {entities_cnt} entities, {rels_cnt} relationships.")
    else:
        logger.info("No root GraphRAG data found for bootstrapping AXI4. Library is initialized but empty.")


def select_protocol(name: str) -> None:
    """Switch the current active database context to the selected protocol."""
    # Ensure it's in our library
    lib = get_library_data()
    if name not in lib:
        raise ValueError(f"Protocol '{name}' is not in the library list.")

    protocol_dir = os.path.join(PROTOCOLS_DIR, name)
    os.makedirs(os.path.join(protocol_dir, "graph_cache"), exist_ok=True)
    os.makedirs(os.path.join(protocol_dir, "vector_index"), exist_ok=True)

    # Re-route CONFIG paths to protocol subdirectories
    CONFIG.graph_cache_dir = os.path.abspath(os.path.join(protocol_dir, "graph_cache"))
    CONFIG.vector_index_dir = os.path.abspath(os.path.join(protocol_dir, "vector_index"))

    # If Neo4j is enabled, set the database to protocol-specific
    if CONFIG.neo4j.enabled:
        # neo4j database names must conform to: [a-zA-Z][a-zA-Z0-9-.]* and be 3-63 chars
        safe_name = "".join(c for c in name.lower() if c.isalnum() or c in "-.")
        CONFIG.neo4j.database = f"protocol-{safe_name}"

    # Reset core singletons so they re-initialize and reload caches from the new folders
    reset_vector_store()
    reset_graph_store()
    reset_registry()

    # Track usage in library
    if lib[name]["status"] == "Ready":
        lib[name]["reuse_count"] = lib[name].get("reuse_count", 0) + 1
        save_library_data(lib)

    logger.info(f"Switched active protocol to '{name}'. DB paths redirected.")


def delete_protocol(name: str) -> None:
    """Clear all data for the protocol and reset status to 'Not Uploaded'."""
    lib = get_library_data()
    if name not in lib:
        return

    protocol_dir = os.path.join(PROTOCOLS_DIR, name)
    if os.path.exists(protocol_dir):
        try:
            shutil.rmtree(protocol_dir)
        except Exception as e:
            logger.error(f"Failed to delete directory {protocol_dir}: {e}")

    lib[name] = {
        "name": name,
        "filename": None,
        "chunks": 0,
        "entities": 0,
        "relationships": 0,
        "created_at": None,
        "reuse_count": 0,
        "status": "Not Uploaded"
    }
    save_library_data(lib)
    logger.info(f"Cleared protocol '{name}' from the library.")
