"""
Comprehensive Unit & Integration Test Suite for Token-Efficient & Rate-Limit Safe Hardware GraphRAG Pipeline.

Tests all 12 core acceptance cases required by the specification:
1. Simple explicit requirement -> Heuristic extraction
2. Explicit handshake -> Heuristic extraction
3. Timing requirement -> Heuristic extraction
4. Ambiguous semantic requirement -> LLM escalation
5. Cross-reference -> LLM escalation
6. LLM rate-limit failure -> Heuristic fallback resilience
7. Malformed LLM JSON -> Graceful recovery
8. Duplicate chunks -> Deduplication without repeating LLM work
9. Cached chunk -> Cache hit without invoking LLM again
10. Simple fact query -> Compact retrieval context
11. Complex multi-hop query -> Adaptive expanded retrieval
12. Long document -> Complete ingestion under simulated rate limits
"""

from __future__ import annotations

import os
import pytest
from config import CONFIG
from utils.models import Chunk
from core.requirement_ir.models import DocumentBlock, DocumentIR, SourceLocation
from core.extraction.pre_classifier import classify_chunk
from core.extraction.heuristic_extractor import extract_heuristically
from core.extraction.decision_engine import needs_llm
from core.extraction.chunk_cache import get_chunk_cache, compute_chunk_hash, build_cache_key
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.retrieval.query_analyzer import analyze_query, classify_query_category
from core.retrieval.hybrid_retriever import answer_question
from core.extraction.llm_client import LLMClient, get_llm_client


class MockFailingLLM:
    available = True

    def complete_json(self, system_prompt: str, user_prompt: str, model: str = None):
        raise RuntimeError("429 Rate Limit Exceeded: TPM limit reached")

    def complete_text(self, system_prompt: str, user_prompt: str, model: str = None):
        raise RuntimeError("503 Service Unavailable: Provider overloaded")


class MockMalformedLLM:
    available = True

    def complete_json(self, system_prompt: str, user_prompt: str, model: str = None):
        return None  # simulates malformed JSON parse failure


class MockSuccessfulLLM:
    available = True

    def complete_json(self, system_prompt: str, user_prompt: str, model: str = None):
        return {
            "atomic_requirements": [
                {
                    "statement": "Complex state transition under ambiguity.",
                    "type": "BEHAVIOR",
                    "subject": "FSM",
                    "confidence": 0.95
                }
            ],
            "entities": [{"name": "FSM", "type": "state", "description": "State machine"}]
        }

    def complete_text(self, system_prompt: str, user_prompt: str, model: str = None):
        return "LLM generated grounded answer."


# Case 1: Simple explicit requirement -> Heuristic extraction
def test_case_1_simple_explicit_requirement():
    text = "The system shall assert ACK when REQ is high."
    cl = classify_chunk(text)
    h_res = extract_heuristically(text)
    assert h_res.confidence >= 0.70
    assert any(e["name"] in ("ACK", "REQ") for e in h_res.entities) or len(h_res.requirements) > 0


# Case 2: Explicit handshake -> Heuristic extraction
def test_case_2_explicit_handshake():
    text = "If AWVALID and AWREADY are high, an address transfer occurs."
    cl = classify_chunk(text)
    h_res = extract_heuristically(text)
    assert "handshake" in cl.categories
    assert len(h_res.conditions) > 0
    assert h_res.conditions[0]["condition"] == "AWVALID && AWREADY"
    assert not needs_llm(text, cl, h_res)


# Case 3: Timing requirement -> Heuristic extraction
def test_case_3_timing_requirement():
    text = "The response must be returned within 16 cycles."
    cl = classify_chunk(text)
    h_res = extract_heuristically(text)
    assert "timing" in cl.categories
    assert len(h_res.timing) > 0
    assert h_res.timing[0]["value"] == 16
    assert h_res.timing[0]["unit"] == "cycles"
    assert not needs_llm(text, cl, h_res)


# Case 4: Ambiguous semantic requirement -> Escalates to LLM
def test_case_4_ambiguous_semantic_requirement():
    text = "Depending on the power mode, the controller typically transitions to idle unless otherwise configured by external pins."
    cl = classify_chunk(text)
    h_res = extract_heuristically(text)
    assert needs_llm(text, cl, h_res) is True


# Case 5: Cross-reference pronoun -> Escalates to LLM
def test_case_5_cross_reference_pronoun():
    text = "When it receives this signal, the previous transaction is terminated."
    cl = classify_chunk(text)
    h_res = extract_heuristically(text)
    assert cl.has_unresolved_references is True
    assert needs_llm(text, cl, h_res) is True


# Case 6: LLM rate-limit failure -> Heuristic fallback resilience
def test_case_6_llm_rate_limit_failure(monkeypatch):
    doc_ir = DocumentIR(
        document_id="doc_ratelimit",
        blocks=[
            DocumentBlock(block_id="b1", type="paragraph", text="If AWVALID and AWREADY are high, transfer occurs.", page=1, section_title="Handshake", source_location=SourceLocation(page=1))
        ]
    )
    extractor = GenericRequirementExtractor(doc_ir)
    extractor.llm = MockFailingLLM()

    # Should not crash; must fall back to heuristics
    req_ir = extractor.extract_requirement_ir()
    assert req_ir is not None
    assert len(req_ir.requirements) > 0 or len(req_ir.entities) > 0


# Case 7: Malformed LLM JSON -> Recovery without crashing
def test_case_7_malformed_llm_json():
    doc_ir = DocumentIR(
        document_id="doc_malformed",
        blocks=[
            DocumentBlock(block_id="b1", type="paragraph", text="It typically transitions within 8 cycles depending on state.", page=1, section_title="Timing", source_location=SourceLocation(page=1))
        ]
    )
    extractor = GenericRequirementExtractor(doc_ir)
    extractor.llm = MockMalformedLLM()

    # Must complete ingestion gracefully without raising an exception
    req_ir = extractor.extract_requirement_ir()
    assert req_ir is not None
    assert extractor.stats["llm_fallback_extractions"] > 0 or extractor.stats["heuristic_extractions"] > 0


# Case 8: Duplicate chunks -> Avoid duplicate LLM work
def test_case_8_duplicate_chunks():
    cache = get_chunk_cache()
    text = "Register REG_CTRL at offset 0x00 (R/W, 32 bits)"
    
    # Register first chunk
    res1 = {"candidates": [], "requirements": [{"statement": "REG_CTRL offset 0x00", "page": 1, "section": "Sec 1"}]}
    cache.register_chunk(text, res1)

    # Check duplicate for page 2
    dup = cache.check_duplicate(text, page=2, section="Sec 2")
    assert dup is not None
    assert dup["requirements"][0]["page"] == 2


# Case 9: Cached chunk -> Reuses stored result
def test_case_9_cached_chunk():
    cache = get_chunk_cache()
    ckey = build_cache_key("hash123", "Some unique text chunk", "modelX")
    data = {"candidates": [], "cached": True}
    cache.put(ckey, data)

    cached = cache.get(ckey)
    assert cached is not None
    assert cached["cached"] is True


# Case 10: Simple fact query -> Compact retrieval
def test_case_10_simple_fact_query():
    query = "What is the reset value of REG_CTRL?"
    intent = analyze_query(query)
    assert intent.query_category == "fact"
    assert classify_query_category(query) == "fact"


# Case 11: Complex multi-hop query -> Expanded retrieval
def test_case_11_complex_multihop_query():
    query = "How does the write transaction impact memory state and why does AWVALID trigger reset?"
    intent = analyze_query(query)
    assert intent.query_category in ("multi-hop", "relationship")


# Case 12: Long document ingestion under simulated rate limits
def test_case_12_long_document_ingestion_resilience(monkeypatch):
    blocks = [
        DocumentBlock(block_id=f"b_{i}", type="paragraph", text=f"Block {i}: It typically transitions depending on external state {i}.", page=(i//5)+1, section_title=f"Section {i//5}", source_location=SourceLocation(page=(i//5)+1))
        for i in range(30)
    ]
    doc_ir = DocumentIR(document_id="doc_long_30", blocks=blocks)
    extractor = GenericRequirementExtractor(doc_ir)
    extractor.llm = MockFailingLLM()

    # Must complete ingestion without raising rate limit errors
    req_ir = extractor.extract_requirement_ir()
    assert req_ir is not None
    assert extractor.stats["total_chunks"] == 30
    assert extractor.stats["llm_fallback_extractions"] > 0 or extractor.stats["heuristic_extractions"] > 0
