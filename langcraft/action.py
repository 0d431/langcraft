from typing import Type, List, Optional, Any, Union, Dict
from pydantic import BaseModel, Field
import time
import concurrent.futures

"""
Module for defining the base classes for actions.
"""


#################################################
class ActionBrief(BaseModel):
    """
    Base class for action briefs, specifying the arguments of an action.
    """

    @classmethod
    def to_schema(cls) -> dict:
        """
        Returns the schema for the action brief.
        """

        def is_optional(python_type: Any) -> bool:
            return (
                hasattr(python_type, "__origin__")
                and python_type.__origin__ is Union
                and len(python_type.__args__) == 2
                and python_type.__args__[1] == type(None)
            )

        def get_type(python_type: Any) -> str:
            if python_type == str:
                return "string"
            elif python_type == int:
                return "integer"
            elif python_type == float:
                return "number"
            elif python_type == bool:
                return "boolean"
            elif is_optional(python_type):
                return get_type(python_type.__args__[0])
            elif hasattr(python_type, "__origin__") and python_type.__origin__ == list:
                return "array"
            return "object"

        def get_items_type(python_type: Any) -> str:
            if hasattr(python_type, "__args__") and len(python_type.__args__) == 1:
                return get_type(python_type.__args__[0])
            return "object"

        schema = {"type": "object", "properties": {}, "required": []}

        for field_name, field_info in cls.model_fields.items():
            field_schema = {
                "type": get_type(field_info.annotation),
                "description": field_info.description or "",
            }

            if field_schema["type"] == "array":
                field_schema["items"] = {"type": get_items_type(field_info.annotation)}

            schema["properties"][field_name] = field_schema

            if field_info.is_required and not is_optional(field_info.annotation):
                schema["required"].append(field_name)

        return schema


#################################################
class ActionResult(BaseModel):
    """
    Base class for action results, providing the output of an action.
    """

    result: str = Field(description="The string representation of the action result.")

    success: bool = Field(
        description="True if the action was successful, False otherwise.", default=True
    )
    error_message: Optional[str] = Field(
        description="The error message if the action failed.", default=None
    )
    cost: Optional[float] = Field(description="The cost in USD.", default=None)
    latency: Optional[float] = Field(
        description="The execution time in milliseconds.", default=None
    )


#################################################
class Action:
    """
    Base class for actions, providing a common interface to run actions.
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        thread_pool_size: int = 5,
    ):
        """
        Initializes an instance of the Action class.

        Args:
            max_batch_size (int, optional): The maximum batch size for processing. Defaults to 1.
            thread_pool_size (int, optional): The number of threads in the thread pool. Defaults to 5.
        """
        self.max_batch_size = max_batch_size
        self.thread_pool_size = thread_pool_size

    def run(self, _brief: ActionBrief) -> ActionResult:
        """
        Executes the action.

        Args:
            _brief (ActionBrief): The brief information about the action.

        Returns:
            ActionResult: The result of the action execution.
        """
        self._preprocess([_brief])

        results = [self._wrap_run_one(_brief)]

        self._postprocess(results)

        return results[0]

    def run_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of actions.

        Args:
            briefs (List[ActionBrief]): A list of action briefs.

        Returns:
            List[ActionResult]: A list of action results.
        """
        self._preprocess(briefs)

        # first chop the list of briefs into batches of max_batch_size
        batches = [
            briefs[i : i + self.max_batch_size]
            for i in range(0, len(briefs), self.max_batch_size)
        ]

        # if just one batch, run it
        if len(batches) == 1:
            return self._wrap_run_one_batch(batches[0])

        # more than one batch, run them in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread_pool_size
        ) as executor:
            results = list(executor.map(self._wrap_run_one_batch, batches))

        # flatten the list of lists
        results = [result for batch in results for result in batch]

        self._postprocess(results)

        return results

    def _preprocess(self, _briefs: List[ActionBrief]):
        """
        Called on a brief before running it to perform any necessary preprocessing.
        """
        pass

    def _postprocess(self, _results: List[ActionResult]):
        """
        Called on a result after running it to perform any necessary postprocessing.
        """
        pass

    def _wrap_run_one(self, brief: ActionBrief) -> ActionResult:
        """
        Runs a single action based on the provided ActionBrief and times execution.
        """
        start = time.time()
        result = self._run_one(brief)
        result.latency = (time.time() - start) * 1000

        return result

    def _run_one(self, brief: ActionBrief) -> ActionResult:
        """
        Runs a batch of up to max_batch_size actions.

        Args:
            briefs (List[ActionBrief]): A list of action briefs.

        Returns:
            List[ActionResult]: A list of action results.
        """
        return self._run_one_batch([brief])[0]

    def _wrap_run_one_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of actions based on the provided ActionBriefs and times execution.
        """
        start = time.time()
        results = self._run_one_batch(briefs)
        for result in results:
            result.latency = (time.time() - start) * 1000

        return results

    def _run_one_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of up to max_batch_size actions.

        Args:
            briefs (List[ActionBrief]): A list of action briefs.

        Returns:
            List[ActionResult]: A list of action results.
        """
        return [self._run_one(brief) for brief in briefs]


#################################################
class ActionDescriptor(BaseModel):
    """
    Providing information about the action.
    """

    name: str = Field(description="The name of the action.")
    description: str = Field(description="The description of the action.")
    brief: Type[ActionBrief] = Field(description="The brief class for the action.")
    action: Type[Action] = Field(
        description="The action class implementing the action."
    )
    result: Type[ActionResult] = Field(description="The result class for the action.")


#################################################
class Actions:
    """
    Index for known action descriptors.
    """

    _action_descriptors: Dict[str, ActionDescriptor] = {}

    @classmethod
    def register(
        cls,
        action_descriptor: ActionDescriptor,
    ):
        name = action_descriptor.name

        if name in cls._action_descriptors:
            raise ValueError(f"Action {name} already registered.")

        cls._action_descriptors[name] = action_descriptor

    @classmethod
    def get(cls, name: str) -> ActionDescriptor:
        """
        Returns the action descriptor with the given name.

        Args:
            name (str): The name of the action.

        Returns:
            ActionDescriptor: The action descriptor.
        """
        return cls._action_descriptors[name]

    @classmethod
    def create_brief(cls, name: str, parameters: Dict) -> ActionBrief:
        """
        Create an action brief from the given action name and parameters.

        Args:
            name (str): The name of the action.
            parameters (Dict): The parameters for the action.

        Returns:
            ActionBrief: The created action brief.
        """
        return Actions.get(name).brief(**parameters)
