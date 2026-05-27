"""
Memory-selection methods for MemoryCD evaluation.

V2 ships with only two methods. Add a new file in this package and register it
in `eval_core.build_method()` to introduce another.

- long_context : keep most-recent N items by recency
- rag          : BM25 retrieval over memory items
"""

from .base_method import BaseMethod
from .long_context import LongContextMethod
from .rag import RAGMethod

__all__ = ["BaseMethod", "LongContextMethod", "RAGMethod"]
