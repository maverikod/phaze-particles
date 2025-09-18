#!/usr/bin/env python3
"""
Base model class for particle models.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """
    Base class for all particle models.

    Provides common interface and functionality for particle modeling.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize base model.

        Args:
            name: Model name
            description: Model description
        """
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate model parameters.

        Returns:
            True if parameters are valid, False otherwise
        """
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the model calculation.

        Returns:
            Dictionary containing calculation results
        """
        pass

    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set model parameter.

        Args:
            key: Parameter key
            value: Parameter value
        """
        self.parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get model parameter.

        Args:
            key: Parameter key
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        return self.parameters.get(key, default)

    def get_results(self) -> Dict[str, Any]:
        """
        Get model results.

        Returns:
            Dictionary containing model results
        """
        return self.results.copy()
