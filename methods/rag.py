"""
Retrieval-Augmented Generation (RAG) method with BM25 retrieval.

This method uses BM25 to retrieve the most relevant interaction histories
based on the target item, rather than simply using the most recent histories.

The method pre-computes BM25 embeddings for all memory items in batch for efficiency,
then uses these embeddings to quickly retrieve top-K items for each query.

Features:
- Batch BM25 index building for efficiency
- In-memory caching for same-user queries within a run
- Persistent disk caching for reuse across different tasks/runs on same dataset

Usage:
    # In-memory cache only (default)
    method = RAGMethod(max_memory_items=10)
    
    # With persistent cache (recommended for multiple tasks on same dataset)
    method = RAGMethod(max_memory_items=10, persistent_cache=True, 
                      cache_dir="cache", dataset_name="All_Beauty")

Dependencies:
    pip install rank-bm25
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from .base_method import BaseMethod

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    # Create a placeholder for type checking
    if TYPE_CHECKING:
        from rank_bm25 import BM25Okapi
    else:
        BM25Okapi = None
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")


class RAGMethod(BaseMethod):
    """
    RAG method that retrieves most relevant interactions using BM25.
    
    This method uses batch processing for efficiency:
    1. Pre-computes document representations (tokenized) for all memory items
    2. Builds BM25 index once per user's memory
    3. Uses the index to efficiently retrieve top-K items for each query
    4. Caches the BM25 model to avoid recomputation for the same memory
    
    The batch approach is more efficient than computing BM25 on-the-fly for each query.
    """
    
    def __init__(
        self, 
        max_memory_items: Optional[int] = 10,
        use_cache: bool = True,
        persistent_cache: bool = False,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize RAG method.
        
        Args:
            max_memory_items: Top-K interactions to retrieve (default: 10).
                             If None, retrieves all interactions ranked by relevance.
            use_cache: Whether to use in-memory cache (default: True for efficiency)
            persistent_cache: Whether to use disk-based persistent cache (default: False).
                             When True, BM25 models are saved to disk and reused across runs.
            cache_dir: Directory for persistent cache files (default: "cache")
            dataset_name: Dataset name for persistent cache organization (e.g., "All_Beauty")
            **kwargs: Additional parameters (unused, for compatibility)
        """
        super().__init__(max_memory_items=max_memory_items, **kwargs)
        
        if not BM25_AVAILABLE:
            raise ImportError(
                "RAG method requires rank-bm25. "
                "Install it with: pip install rank-bm25"
            )
        
        self.use_cache = use_cache
        self.persistent_cache = persistent_cache
        self.dataset_name = dataset_name
        
        # Setup cache directory for persistent storage
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("cache")
        
        # In-memory cache: {user_id: (bm25_model, tokenized_corpus, memory_list)}
        self._bm25_cache: Dict[str, Tuple[Any, List[List[str]], List[Dict[str, Any]]]] = {}
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._disk_loads = 0
        self._disk_saves = 0
    
    def _create_document_text(self, interaction: Dict[str, Any]) -> str:
        """
        Create a text representation of an interaction for BM25 indexing.
        
        Args:
            interaction: User interaction dictionary
        
        Returns:
            Text representation combining title, review text, and metadata
        """
        parts = []
        
        # Add title
        if "title" in interaction and interaction["title"]:
            parts.append(interaction["title"])
        
        # Add review text
        if "text" in interaction and interaction["text"]:
            parts.append(interaction["text"])
        
        # Add item metadata if available
        if "item_meta" in interaction:
            meta = interaction["item_meta"]
            if "title" in meta and meta["title"]:
                parts.append(meta["title"])
            if "main_category" in meta and meta["main_category"]:
                parts.append(meta["main_category"])
            if "description" in meta and meta["description"]:
                desc = str(meta["description"])
                if len(desc) > 500:
                    desc = desc[:500]
                parts.append(desc)
        
        return " ".join(parts)
    
    def _create_query_text(
        self, 
        target_item: Dict[str, Any],
        task: str
    ) -> str:
        """
        Create a query text from the target item for BM25 retrieval.
        
        Args:
            target_item: Target item dictionary
            task: Task name (may influence what fields to emphasize)
        
        Returns:
            Query text for BM25 retrieval
        """
        parts = []
        
        # Add title
        if "title" in target_item and target_item["title"]:
            parts.append(target_item["title"])
        
        # For review generation/summarization, we might have review text
        if task in ["review_summarization", "review_generation"]:
            if "text" in target_item and target_item["text"]:
                parts.append(target_item["text"])
        
        # Add item metadata if available
        if "item_meta" in target_item:
            meta = target_item["item_meta"]
            if "title" in meta and meta["title"]:
                parts.append(meta["title"])
            if "main_category" in meta and meta["main_category"]:
                parts.append(meta["main_category"])
            if "description" in meta and meta["description"]:
                desc = str(meta["description"])
                if len(desc) > 500:
                    desc = desc[:500]
                parts.append(desc)
        
        return " ".join(parts)
    
    def _get_cache_filename(self, user_id: str) -> Path:
        """
        Get the cache file path for a given user ID.
        
        Args:
            user_id: User ID
        
        Returns:
            Path to cache file
        """
        # Sanitize user_id for filesystem (replace problematic characters)
        safe_user_id = user_id.replace('/', '_').replace('\\', '_')
        
        if self.dataset_name:
            # Organize by dataset: cache/All_Beauty/user_XXXX.pkl
            cache_file = self.cache_dir / self.dataset_name / f"{safe_user_id}.pkl"
        else:
            # Flat structure: cache/user_XXXX.pkl
            cache_file = self.cache_dir / f"{safe_user_id}.pkl"
        
        return cache_file
    
    def _load_from_disk(self, cache_file: Path) -> Optional[Tuple[Any, List[List[str]], List[Dict[str, Any]]]]:
        """
        Load BM25 model from disk cache.
        
        Args:
            cache_file: Path to cache file
        
        Returns:
            Tuple of (bm25_model, tokenized_corpus, memory_list) or None if not found
        """
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self._disk_loads += 1
                return cached_data
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_file}: {e}")
            return None
    
    def _save_to_disk(
        self, 
        cache_file: Path, 
        bm25_model: Any, 
        tokenized_corpus: List[List[str]], 
        memory: List[Dict[str, Any]]
    ):
        """
        Save BM25 model to disk cache.
        
        Args:
            cache_file: Path to cache file
            bm25_model: BM25 model to save
            tokenized_corpus: Tokenized corpus
            memory: Original memory list
        """
        try:
            # Create directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to disk
            with open(cache_file, 'wb') as f:
                pickle.dump((bm25_model, tokenized_corpus, memory), f)
                self._disk_saves += 1
                
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")
    
    def _build_bm25_index(
        self, 
        memory: List[Dict[str, Any]]
    ) -> Tuple[Any, List[List[str]]]:
        """
        Build BM25 index for a memory list (batch processing).
        
        This method processes all memory items in batch to create the BM25 index,
        which is more efficient than processing items one-by-one.
        
        Args:
            memory: Full list of user's interaction history
        
        Returns:
            Tuple of (BM25 model, tokenized corpus)
        """
        # Batch process: convert all interactions to text
        corpus_texts = [self._create_document_text(inter) for inter in memory]
        
        # Batch tokenize (simple whitespace tokenization)
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        
        # Create BM25 model (this computes IDF scores in batch)
        bm25 = BM25Okapi(tokenized_corpus)
        
        return bm25, tokenized_corpus
    
    def select_memory(
        self,
        memory: List[Dict[str, Any]],
        target_item: Dict[str, Any],
        task: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Select top-K most relevant memory items using BM25 retrieval with caching.
        
        This method uses hierarchical caching for maximum efficiency:
        1. Checks in-memory cache by user_id (fastest)
        2. Checks disk cache if persistent_cache enabled (fast)
        3. Builds BM25 index if not cached (slower, first time only)
        4. Saves to both memory and disk cache
        
        Args:
            memory: Full list of user's interaction history
            target_item: Target item for prediction/generation
            task: Task name (rating_prediction, review_summarization, etc.)
            user_id: User ID for caching (REQUIRED for persistent cache)
        
        Returns:
            Top-K most relevant memory items based on BM25 scores
        """
        if not memory:
            return []
        
        # If memory is smaller than or equal to max_memory_items, return all
        if self.max_memory_items is not None and len(memory) <= self.max_memory_items:
            return memory
        
        # Check if user_id is provided for caching
        if not user_id and (self.use_cache or self.persistent_cache):
            print("Warning: user_id not provided - caching disabled for this query")
        
        bm25 = None
        tokenized_corpus = None
        memory_to_use = memory
        
        # Level 1: Check in-memory cache by user_id (fastest)
        if self.use_cache and user_id and user_id in self._bm25_cache:
            bm25, tokenized_corpus, cached_memory = self._bm25_cache[user_id]
            # Verify memory matches (safety check)
            if len(cached_memory) == len(memory):
                memory_to_use = cached_memory
                self._cache_hits += 1
            else:
                # Memory changed, will rebuild
                bm25 = None
        
        # Level 2: Check disk cache (fast, if in-memory cache missed)
        if bm25 is None and self.persistent_cache and user_id:
            cache_file = self._get_cache_filename(user_id)
            cached_data = self._load_from_disk(cache_file)
            
            if cached_data is not None:
                bm25, tokenized_corpus, cached_memory = cached_data
                # Verify memory matches
                if len(cached_memory) == len(memory):
                    memory_to_use = cached_memory
                    # Also store in in-memory cache for next time
                    if self.use_cache:
                        self._bm25_cache[user_id] = (bm25, tokenized_corpus, cached_memory)
                    self._cache_hits += 1
                else:
                    # Disk cache outdated, will rebuild
                    bm25 = None
        
        # Level 3: Build BM25 index (first time or cache miss)
        if bm25 is None:
            self._cache_misses += 1
            bm25, tokenized_corpus = self._build_bm25_index(memory)
            memory_to_use = memory
            
            # Save to in-memory cache
            if self.use_cache and user_id:
                self._bm25_cache[user_id] = (bm25, tokenized_corpus, memory)
            
            # Save to disk cache
            if self.persistent_cache and user_id:
                cache_file = self._get_cache_filename(user_id)
                self._save_to_disk(cache_file, bm25, tokenized_corpus, memory)
        
        # Create query from target item
        query_text = self._create_query_text(target_item, task)
        tokenized_query = query_text.lower().split()
        
        # Get BM25 scores for all documents (efficient batch scoring)
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        k = self.max_memory_items if self.max_memory_items is not None else len(memory_to_use)
        k = min(k, len(memory_to_use))
        
        # Get indices of top-K scores
        top_k_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        # Sort indices by original order (to maintain temporal order in prompt)
        top_k_indices_sorted = sorted(top_k_indices)
        
        # Return selected memory items
        selected_memory = [memory_to_use[i] for i in top_k_indices_sorted]
        
        return selected_memory
    
    def clear_cache(self, clear_disk: bool = False):
        """
        Clear the BM25 cache to free memory.
        
        Args:
            clear_disk: If True, also delete persistent cache files from disk
        """
        self._bm25_cache.clear()
        
        if clear_disk and self.persistent_cache:
            import shutil
            if self.dataset_name:
                cache_path = self.cache_dir / self.dataset_name
            else:
                cache_path = self.cache_dir
            
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                    print(f"Cleared disk cache: {cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to clear disk cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "memory_cached_users": len(self._bm25_cache),
            "cache_enabled": self.use_cache,
            "persistent_cache_enabled": self.persistent_cache,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "disk_loads": self._disk_loads,
            "disk_saves": self._disk_saves,
        }
        
        # Count disk cache files if persistent cache is enabled
        if self.persistent_cache:
            if self.dataset_name:
                cache_path = self.cache_dir / self.dataset_name
            else:
                cache_path = self.cache_dir
            
            if cache_path.exists():
                disk_files = list(cache_path.glob("*.pkl"))
                stats["disk_cached_users"] = len(disk_files)
            else:
                stats["disk_cached_users"] = 0
        
        # Calculate hit rate
        total_queries = self._cache_hits + self._cache_misses
        if total_queries > 0:
            stats["hit_rate"] = self._cache_hits / total_queries
        else:
            stats["hit_rate"] = 0.0
        
        return stats

