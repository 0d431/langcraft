import os
import json
import glob
import fnmatch
import pkg_resources

"""
This module provides a class that represents the map of known LLMs.
"""


#################################################
class LLMs:
    """A class that represents the map of known LLMs."""

    _model_map = None

    @classmethod
    def _get(cls):
        """
        Retrieves the model map.

        If the model map is not already loaded, it loads the map from JSON files
        in the current directory that match the pattern "models*.json".

        Returns:
            dict: The model map.
        """
        if cls._model_map is None:
            cls._model_map = {}

            file_path = pkg_resources.resource_filename("langcraft", "data/models*.json")
            for model_filename in glob.glob(file_path):
                with open(model_filename, "r") as f:
                    cls._model_map.update(json.load(f))

        return cls._model_map

    @classmethod
    def _get_model_metadata(cls, model_name):
        """
        Return the metadata for the given model name.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            metadata: The model object corresponding to the given model name.

        Raises:
            ValueError: If the model with the given name is not found.
        """
        for name, metadata in LLMs._get().items():
            if fnmatch.fnmatch(name, model_name):
                return metadata

        raise ValueError(f"Model {model_name} not found.")

    @classmethod
    def resolve_model(cls, model_name):
        """
        Returns the best matching model name based on the provided input.

        Args:
            cls (class): The class object.
            model_name (str): The input model name.

        Returns:
            str: The best matching model name.

        Raises:
            ValueError: If the provided model name is not found.
        """
        if model_name is None or model_name.strip() == "":
            # use default LLM
            return os.environ.get("LANGCRAFT_DEFAULT_LLM", "claude-3-haiku-20240307")

        if model_name.startswith("env:"):
            # retrieve the environment variable
            model_name = os.environ.get(model_name[4:], model_name)

        # find best prefix match
        best_match = None
        for name in cls._get().keys():
            if fnmatch.fnmatch(model_name, name) or name.startswith(model_name):
                if best_match is None or len(name) > len(best_match):
                    best_match = name

        if best_match is not None:
            return best_match

        raise ValueError(f"Model {model_name} not found.")

    @classmethod
    def get_context_window_size(cls, model_name: str) -> int:
        """Return the length of the model's context window"""

        return LLMs._get_model_metadata(model_name).get("context_window_size")

    @classmethod
    def get_max_output_tokens(cls, model_name: str) -> int:
        """Return the maximum length of the response tokens"""

        return LLMs._get_model_metadata(model_name).get("max_output_tokens")

    @classmethod
    def get_prompt_cost(cls, model_name: str, number_of_tokens: int) -> float:
        """Return the cost of the prompt in USD"""
        return (
            number_of_tokens
            * LLMs._get_model_metadata(model_name)["prompt_cost"]
            / 1.0e6
        )

    @classmethod
    def get_completion_cost(cls, model_name: str, number_of_tokens: int) -> float:
        """Return the cost of the completion in USD"""
        return (
            number_of_tokens
            * LLMs._get_model_metadata(model_name)["completion_cost"]
            / 1.0e6
        )

    @classmethod
    def get_embedding_cost(cls, model_name: str, number_of_tokens: int) -> float:
        """Return the cost of the prompt in USD"""
        return (
            number_of_tokens
            * LLMs._get_model_metadata(model_name)["embedding_cost"]
            / 1.0e6
        )