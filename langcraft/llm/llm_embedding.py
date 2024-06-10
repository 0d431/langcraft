from pydantic import Field
from fnmatch import fnmatch
from typing import Optional, List, Dict
from langcraft.action import *
from langcraft.action import ActionBrief, ActionResult
from langcraft.utils import extract_tag
from langcraft.llm.llm_models import LLMs

"""
Module for defining the classes for embedding actions.
"""


#################################################
class EmbeddingBrief(ActionBrief):
    """
    Class for an embedding brief.
    """

    model_name: Optional[str] = Field(
        description="The name of the model to use.", default=None
    )

    dimensions: Optional[int] = Field(
        description="The number of dimensions for the embedding.", default=None
    )

    text: str = Field(description="The text to embed.")

    class Config:
        # to prevent complaints about the model_name field
        protected_namespaces = ()


#################################################
class EmbeddingResult(ActionResult):
    """
    Represents the result of an embedding action.
    """

    model_name: str = Field(description="The name of the model used for embedding.")

    embedding: Optional[List[float]] = Field(
        description="The embedding of the text.", default=None
    )

    tokens: Optional[int] = Field(
        description="The number of tokens embedded.", default=None
    )

    class Config:
        # to prevent complaints about the model_name field
        protected_namespaces = ()


#################################################
class EmbeddingAction(Action):
    """
    An action that delegates to an LLM to compute embeddings.
    """

    @classmethod
    def get_descriptor(cls) -> ActionDescriptor:
        return ActionDescriptor(
            name="llm-embedding",
            description="Generates embeddings for text.",
            brief=EmbeddingBrief,
            action=cls,
            result=EmbeddingResult,
        )

    # dictionary of model name matchers to implementations
    _embedding_action_implementations: Dict[str, Action] = {}

    @classmethod
    def register_implementation(
        cls, model_name_matchers: List[str], implementation: Action
    ):
        """
        Register an implementation for a given model name matcher.

        Args:
            cls (type): The class to register the implementation for.
            model_name_matchers (List[str]): A list of model name matchers to register.
            implementation (LanguageAction): The implementation to register.

        Raises:
            ValueError: If a model name matcher is already registered for the given class.

        """
        for model_name_matcher in model_name_matchers:
            if model_name_matcher in cls._embedding_action_implementations:
                raise ValueError(
                    f"Model matcher already registered: {model_name_matcher}"
                )

            cls._embedding_action_implementations[model_name_matcher] = implementation

    @classmethod
    def _get_implementation(cls, model_name: str):
        """
        Get the implementation for the given model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            implementation: The implementation associated with the model name.

        Raises:
            ValueError: If the model name is not found.
        """
        for (
            model_name_matcher,
            implementation,
        ) in cls._embedding_action_implementations.items():
            if fnmatch(model_name, model_name_matcher):
                return implementation

        raise ValueError(f"Model not found: {model_name}")

    def run(self, brief: ActionBrief) -> ActionResult:
        """
        Executes the action specified by the given ActionBrief.

        Args:
            brief (ActionBrief): The ActionBrief object containing the necessary information for the action.

        Returns:
            ActionResult: The result of executing the action.
        """
        implementation = EmbeddingAction._get_implementation(brief.model_name)

        return implementation().run(brief)

    def run_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of action briefs.

        Args:
            briefs (List[ActionBrief]): A list of action briefs.

        Returns:
            List[ActionResult]: A list of action results.
        """
        implementation = EmbeddingAction._get_implementation(briefs[0].model_name)

        return implementation().run_batch(briefs)


#################################################
Actions.register(EmbeddingAction.get_descriptor())


#################################################
class EmbeddingDelegateAction(Action):
    """
    Base class for actual embedding implementations.
    """

    def _preprocess(self, briefs: List[ActionBrief]):
        """
        Preprocesses the given list of action briefs to resolve the model name if needed.

        Args:
            briefs (List[ActionBrief]): The list of action briefs to be preprocessed.

        Raises:
            ValueError: If an invalid brief type is encountered.
        """
        # get the model and max_tokens settings
        model_name = None
        for brief in briefs:
            if isinstance(brief, EmbeddingBrief):
                brief.model_name = LLMs.resolve_model(brief.model_name)

                if not model_name:
                    model_name = brief.model_name
                elif model_name != brief.model_name:
                    raise ValueError(
                        f"Model mismatch in batch: {model_name} != {brief.model_name}"
                    )

                if LLMs.resolve_model(brief.model_name) is None:
                    raise ValueError(f"Model not found: {brief.model_name}")
            else:
                raise ValueError(
                    f"Invalid brief type: {type(brief)}; expected EmbeddingBrief."
                )

        super()._preprocess(briefs)

    def _postprocess(self, results: List[ActionResult]):
        """
        Postprocesses the results by calculating the cost for each result.

        Args:
            results (List[ActionResult]): The list of results to be postprocessed.
        """
        # calculate the cost & log
        for result in results:
            if isinstance(result, EmbeddingResult):
                result.cost = LLMs.get_embedding_cost(result.model_name, result.tokens)

        super()._postprocess(results)
