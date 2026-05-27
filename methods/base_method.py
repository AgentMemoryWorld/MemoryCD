"""
Base class for prompting methods.

This module defines the abstract base class that all prompting methods
(long-context, RAG, etc.) should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseMethod(ABC):
    """
    Abstract base class for prompting methods.
    
    All method implementations should inherit from this class and implement
    the select_memory method to define how interaction histories are selected
    for LLM prompting.
    """
    
    def __init__(
        self, 
        max_memory_items: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize base method.
        
        Args:
            max_memory_items: Maximum number of memory items to select
            **kwargs: Additional method-specific parameters
        """
        self.max_memory_items = max_memory_items
    
    @abstractmethod
    def select_memory(
        self,
        memory: List[Dict[str, Any]],
        target_item: Dict[str, Any],
        task: str
    ) -> List[Dict[str, Any]]:
        """
        Select relevant memory items for prompting based on target item and task.
        
        Args:
            memory: Full list of user's interaction history
            target_item: Target item for prediction/generation
            task: Task name (rating_prediction, review_summarization, etc.)
        
        Returns:
            Selected memory items to include in the prompt
        """
        pass
    
    def get_method_name(self) -> str:
        """Return the name of this method."""
        return self.__class__.__name__.replace('Method', '').lower()

