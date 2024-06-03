import os
import json
import glob
import fnmatch


#################################################
class LLMs:
    """A class that represents the map of known LLMs."""

    _model_map = None

    @classmethod
    def _get(cls):
        """
        Returns the Anthropic client instance.

        If the client instance is not already created, it will be created based on the environment variables.

        Returns:
            The client instance.

        Raises:
            KeyError: If the required environment variables are not set.
        """
        if cls._model_map is None:
            cls._model_map = {}

            for model_filename in glob.glob("./models*.json"):
                with open(model_filename, "r") as f:
                    cls._model_map.update(json.load(f))

        return cls._model_map

    @classmethod
    def get_model_name(cls, model_name):
        if model_name is None or model_name.strip() == "":
            # Use the default LLM
            return os.environ.get("LANGCRAFT_DEFAULT_LLM", "claude-3-haiku-20240307")

        if model_name.startswith("env:"):
            model_name = os.environ.get(model_name[4:], model_name)

        # find the best prefix match
        best_match = None
        for name in cls._get().keys():
            if fnmatch.fnmatch(model_name, name) or name.startswith(model_name):
                if best_match is None or len(name) > len(best_match):
                    best_match = name

        if best_match is not None:
            return best_match

        raise ValueError(f"Model {model_name} not found.")

    @classmethod
    def get_model(cls, model_name):
        """
        Return the model for the given model name.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            model: The model object corresponding to the given model name.

        Raises:
            ValueError: If the model with the given name is not found.
        """
        for name, model in LLMs._get().items():
            if fnmatch.fnmatch(name, model_name):
                return model

        raise ValueError(f"Model {model_name} not found.")

    @classmethod
    def get_context_window_size(cls, model_name: str) -> int:
        """Return the length of the model's context window"""

        return LLMs.get_model(model_name).get("context_window_size")

    @classmethod
    def get_max_output_tokens(cls, model_name: str) -> int:
        """Return the maximum length of the response tokens"""

        return LLMs.get_model(model_name).get("max_output_tokens")

    @classmethod
    def get_prompt_cost(cls, model_name: str, number_of_tokens: int) -> float:
        """Return the cost of the prompt in USD"""
        return number_of_tokens * LLMs.get_model(model_name)["prompt_cost"] / 1.0e6

    @classmethod
    def get_completion_cost(cls, model_name: str, number_of_tokens: int) -> float:
        """Return the cost of the completion in USD"""
        return number_of_tokens * LLMs.get_model(model_name)["completion_cost"] / 1.0e6
