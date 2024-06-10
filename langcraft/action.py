from typing import Type, List, Optional, Any, Union, Dict
from pydantic import BaseModel, Field, create_model
import inspect
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

            if field_info.is_required():
                schema["required"].append(field_name)

        return schema


#################################################
class ActionResult(BaseModel):
    """
    Base class for action results, providing the output of an action.
    """

    success: bool = Field(
        description="True if the action was successful, False otherwise.", default=True
    )

    error_message: Optional[str] = Field(
        description="The error message if the action failed.", default=None
    )

    cost: Optional[float] = Field(description="The cost in USD.", default=0.0)

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

    def get_name(self):
        """
        Returns the name of the action.
        """
        return self.__class__.__name__

    def _preprocess(self, _briefs: List[ActionBrief]):
        """
        Called on a brief before running it to perform any necessary preprocessing.
        """
        print(f"Running action {self.get_name()}")

    def _postprocess(self, _results: List[ActionResult]):
        """
        Called on a result after running it to perform any necessary postprocessing.
        """
        pass

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

    def _wrap_run_one(self, brief: ActionBrief) -> ActionResult:
        """
        Runs a single action based on the provided ActionBrief and times execution.
        """
        start = time.time()

        try:
            result = self._run_one(brief)
        except Exception as e:
            result = ActionResult(
                success=False,
                error_message=str(e),
            )

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

        # run batch(es)
        if len(batches) == 1:
            results = self._wrap_run_one_batch(batches[0])
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.thread_pool_size
            ) as executor:
                results = list(executor.map(self._wrap_run_one_batch, batches))

            results = [result for batch in results for result in batch]

        self._postprocess(results)

        return results

    def _wrap_run_one_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of actions based on the provided ActionBriefs and times execution.
        """
        start = time.time()

        try:
            results = self._run_one_batch(briefs)
        except Exception as e:
            results = [
                ActionResult(
                    success=False,
                    error_message=str(e),
                )
                for _ in briefs
            ]

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
    Describes an the action.
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
    Index for known actions.
    """

    _action_descriptors: Dict[str, ActionDescriptor] = {}

    @classmethod
    def register(cls, action_descriptor: ActionDescriptor):
        """
        Register a new action descriptor.

        Args:
            cls (class): The class to register the action descriptor for.
            action_descriptor (ActionDescriptor): The action descriptor to register.

        Raises:
            ValueError: If the action descriptor is already registered.
        """
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

    @classmethod
    def generate_action(cls, func: Any) -> ActionDescriptor:
        """
        Generate an Action class dynamically based on the provided function.

        Args:
            cls (Type[Action]): The base class for the generated Action class.
            func (Any): The function to generate the Action class from.

        Returns:
            ActionDescriptor: The action descriptor for the generated Action class.
        """
        # get parameters from the function signature
        parameters = inspect.signature(func).parameters

        # extract descriptions from the docstring
        descriptions = {}
        doc_lines = func.__doc__
        if "Args:" in doc_lines:
            doc_lines = doc_lines.split("Args:")[1].split("\n")
            for param in parameters.values():
                if param.name != "return":
                    for line in doc_lines:
                        if line.strip().startswith(param.name):
                            descriptions[param.name] = line.split(":")[1].strip()
                            break

        # create fields for the Pydantic model considering optional parameters with defaults
        fields = {
            param.name: Field(
                default=(
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else ...
                ),
                description=descriptions.get(param.name, ""),
            )
            for param in parameters.values()
            if param.name != "return"
        }

        # create the brief class dynamically
        brief_class_name = "DynamicBrief"
        brief_fields = {
            param_name: (
                param.annotation,
                Field(
                    description=fields[param_name].description,
                    default=fields[param_name].default,
                ),
            )
            for param_name, param in parameters.items()
        }

        BriefClass = create_model(
            brief_class_name, **brief_fields, __base__=ActionBrief
        )

        # create a result class dynamically
        result_class_name = "DynamicResult"
        return_annotation = inspect.signature(func).return_annotation
        result_fields = {
            "result": (
                (
                    str
                    if return_annotation == inspect.Signature.empty
                    else return_annotation
                ),
                Field(description=f"The result of {func.__name__}()."),
            ),
        }

        ResultClass = create_model(
            result_class_name, **result_fields, __base__=ActionResult
        )

        # define a dynamic action class
        class DynamicAction(Action):
            NAME = func.__name__

            @classmethod
            def get_descriptor(cls) -> ActionDescriptor:
                return ActionDescriptor(
                    name=cls.NAME,
                    description="\n".join(doc_lines)
                    .split("Args:")[0]
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .strip(),
                    brief=BriefClass,
                    action=cls,
                    result=ResultClass,
                )

            def __init__(self):
                super().__init__(name=DynamicAction.NAME)

            def _run_one(self, brief: Any) -> ActionResult:
                func_args = {field: getattr(brief, field) for field in fields}
                return ResultClass(result=func(**func_args))

        # register action descriptor
        action_descriptor = DynamicAction.get_descriptor()
        Actions.register(action_descriptor)

        return action_descriptor
