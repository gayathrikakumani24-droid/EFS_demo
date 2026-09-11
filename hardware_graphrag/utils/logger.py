"""
Centralized logging for the GraphRAG pipeline.

All modules pull a logger from `get_logger(name)`. Logs are written to both
the console (for `streamlit run` terminal output) and to a rotating file
under data/logs/, and also mirrored into an in-memory ring buffer so the
Streamlit "Logs" page can display them live without re-reading the file
on every rerun.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Deque, Dict, List

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

_MAX_BUFFER = 2000
_LOG_BUFFER: Deque[Dict[str, str]] = deque(maxlen=_MAX_BUFFER)


class BufferHandler(logging.Handler):
    """Keeps the most recent N log records in memory, categorized by logger name."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            category = _categorize(record.name)
            _LOG_BUFFER.append({
                "time": self.formatter.formatTime(record) if self.formatter else "",
                "level": record.levelname,
                "logger": record.name,
                "category": category,
                "message": record.getMessage(),
                "formatted": msg,
            })
        except Exception:
            self.handleError(record)


def _categorize(logger_name: str) -> str:
    name = logger_name.lower()
    if "pars" in name:
        return "parsing"
    if "chunk" in name:
        return "chunking"
    if "extract" in name:
        return "extraction"
    if "graph" in name or "neo4j" in name:
        return "graph"
    if "vector" in name or "faiss" in name or "embed" in name:
        return "vector"
    if "retriev" in name or "qa" in name:
        return "retrieval"
    return "general"


_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
_formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    root = logging.getLogger("graphrag")
    if root.handlers:
        _root_configured = True
        return
    if _root_configured:
        return
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setFormatter(_formatter)
    console.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(_formatter)
    file_handler.setLevel(logging.DEBUG)

    buffer_handler = BufferHandler()
    buffer_handler.setFormatter(_formatter)
    buffer_handler.setLevel(logging.DEBUG)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(buffer_handler)
    root.propagate = False
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger('graphrag.parsing.pdf')."""
    _configure_root()
    full_name = name if name.startswith("graphrag") else f"graphrag.{name}"
    return logging.getLogger(full_name)


def get_recent_logs(category: str = "all", level: str = "ALL", limit: int = 500) -> List[Dict[str, str]]:
    """Fetch recent log records for the Streamlit Logs page."""
    items = list(_LOG_BUFFER)[-limit:]
    if category != "all":
        items = [i for i in items if i["category"] == category]
    if level != "ALL":
        items = [i for i in items if i["level"] == level]
    return list(reversed(items))


def clear_logs() -> None:
    _LOG_BUFFER.clear()
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.truncate(0)
        except Exception:
            pass


def _load_initial_logs() -> None:
    if not os.path.exists(LOG_FILE):
        return
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[-_MAX_BUFFER:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" | ", 3)
                if len(parts) == 4:
                    asctime, levelname, logger_name, message = parts
                    levelname = levelname.strip()
                    logger_name = logger_name.strip()
                    category = _categorize(logger_name)
                    _LOG_BUFFER.append({
                        "time": asctime,
                        "level": levelname,
                        "logger": logger_name,
                        "category": category,
                        "message": message,
                        "formatted": line,
                    })
    except Exception:
        pass


# Load logs from file on startup to populate the in-memory ring buffer
_load_initial_logs()
