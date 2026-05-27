"""
Long-context prompting method.

This method uses direct long-context prompting where all (or the most recent N)
interaction histories are directly concatenated into the prompt.

Usage:
    method = LongContextMethod(max_memory_items=50)
    selected_memory = method.select_memory(memory, target_item, task="rating_prediction")
"""

from typing import Dict, List, Any, Optional
from .base_method import BaseMethod


class LongContextMethod(BaseMethod):
    """
    Long-context prompting method that selects the most recent N interactions.
    
    This is the default/baseline approach where interaction histories are
    selected based on recency (timestamp) and included in full in the prompt.
    """
    
    def __init__(self, max_memory_items: Optional[int] = None, **kwargs):
        """
        Initialize long-context method.
        
        Args:
            max_memory_items: Maximum number of recent memory items to include.
                             If None, includes all memory items.
            **kwargs: Additional parameters (unused, for compatibility)
        """
        super().__init__(max_memory_items=max_memory_items, **kwargs)
    
    def select_memory(
        self,
        memory: List[Dict[str, Any]],
        target_item: Dict[str, Any],
        task: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Select most recent memory items for prompting.
        
        Args:
            memory: Full list of user's interaction history (assumed sorted by timestamp)
            target_item: Target item for prediction/generation (unused in this method)
            task: Task name (unused in this method)
            user_id: User ID (unused in this method, but required for interface consistency)
        
        Returns:
            Most recent N memory items (or all if max_memory_items is None)
        """
        if self.max_memory_items is None or len(memory) <= self.max_memory_items:
            return memory
        
        # Select the most recent N items
        return memory[-self.max_memory_items:]

