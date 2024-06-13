from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TypedDict
import typing

from weblinx.utils.format import format_output_dictionary

# Custom types
UID = typing.NewType("UID", str)
AttrsCore = TypedDict(
    "AttrsCore",
    {"class": str, "title": str, "href": str, "aria-label": str, "d": str, "src": str},
)


class BBox(TypedDict):
    """
    A class to represent the bounding box of an element.

    Attributes
    ----------
    x : int
        The x-coordinate of the bounding box.
    y : int
        The y-coordinate of the bounding box.
    width : float
        The width of the bounding box.
    height : float
        The height of the bounding box.
    top : float, optional
        The top position of the bounding box, calculated from `y` if not provided.
    bottom : float, optional
        The bottom position of the bounding box, calculated from `y` and `height` if not provided.
    left : float, optional
        The left position of the bounding box, calculated from `x` if not provided.
    right : float, optional
        The right position of the bounding box, calculated from `x` and `width` if not provided.
    """
    x: int
    y: int
    width: float
    height: float
    top: float = None
    bottom: float = None
    left: float = None
    right: float = None

    def __post_init__(self):
        """
        Ensures required attributes are provided and calculates optional attributes if not given.
        For example, if `top` is not provided, it is calculated from `y`.
        """
        if any(x is None for x in [self.x, self.y, self.width, self.height]):
            raise ValueError("x, y, width, and height must be provided.")

        if self.top is None:
            self.top = self.y

        if self.bottom is None:
            self.bottom = self.y + self.height

        if self.left is None:
            self.left = self.x

        if self.right is None:
            self.right = self.x + self.width


@dataclass
class State:
    """
    A class to represent the state during navigation.

    Attributes
    ----------
    index : int
        The index of the state in the sequence of states.
    html : str
        The DOM tree represented using HTML.
    bboxes : Dict[UID, BBox]
        A dictionary mapping unique IDs to bounding boxes.
    viewport_height : int
        The height of the viewport of the browser.
    viewport_width : int
        The width of the viewport of the browser.
    type : str
        The type of the state, either "browser" or "chat".

    Methods
    -------
    from_dict(cls, dictionary):
        Creates a `State` instance from a dictionary.
    to_dict():
        Converts the `State` instance to a dictionary.
    """
    index: int
    html: str
    bboxes: Dict[UID, BBox]
    viewport_height: int
    viewport_width: int
    type: str  # either "browser" or "chat"

    # check type
    def __post_init__(self):
        if self.type not in ["browser", "chat"]:
            raise ValueError("type must be either 'browser' or 'chat'.")

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a `State` instance from a dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary to create the `State` instance from.

        Returns
        -------
        State
            The created `State` instance.
        """
        return cls(
            index=dictionary["index"],
            html=dictionary["html"],
            bboxes=dictionary["bboxes"],
            viewport_height=dictionary["viewport_height"],
            viewport_width=dictionary["viewport_width"],
            type=dictionary["type"],
        )

    def to_dict(self):
        """
        Converts the `State` instance to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the `State` instance.
        """
        return {
            "index": self.index,
            "html": self.html,
            "bboxes": self.bboxes,
            "viewport_height": self.viewport_height,
            "viewport_width": self.viewport_width,
            "type": self.type,
        }

@dataclass
class Action:
    """
    A class to represent an action taken by the user.

    Attributes
    ----------
    type : str
        The type of the action, either "chat" or "browser".
    index : int
        The index of the action in the sequence of state/actions.
    intent : str
        The intent of the action (e.g., "click", "type", "scroll", "say").
    args : Dict[str, str]
        A dictionary of arguments associated with the action, such as the unique
        ID of the element clicked, the text typed, or the message said.
    timestamp : float
        The timestamp of the action in seconds, relative to the start time.
    tag : str, optional
        The HTML tag associated with the action (e.g., "button", "input").
    attrs : AttrsCore, optional
        The attributes associated with the action (e.g., "class", "title", "href", "aria-label", "d", "src").
    """
    type: str
    index: int
    intent: str
    args: Dict[str, str]
    timestamp: float
    tag: str = None
    attrs: AttrsCore = None

    def get(self, key):
        """
        Retrieves the value of the specified argument key.

        Parameters
        ----------
        key : str
            The key of the argument to retrieve.

        Returns
        -------
        str
            The value of the specified argument key.
        """
        return self.args.get(key, None)

    @classmethod
    def from_dict(
        cls,
        dictionary: Dict,
        included_attrs: Tuple[str] = ("class", "title", "href", "aria-label", "d", "src"),
    ) -> "Action":
        """
        Creates an `Action` instance from a dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary to create the `Action` instance from. It should have the following
            keys: "intent", "index", "timestamp", "attrs" (optional), "tag" (optional), and 
            any other keys as arguments. Moreover, the type of the action is inferred from 
            the "intent" key.
        included_attrs : tuple of str, optional
            A tuple of attribute keys to include in the `attrs` dictionary.

        Returns
        -------
        Action
            The created `Action` instance.
        """
        di = deepcopy(dictionary)
        intent = di.pop("intent")
        index = di.pop("index")
        timestamp = di.pop("timestamp")
        attrs = di.pop("attrs", None)
        if attrs is not None:
            attrs = {k: v for k, v in attrs.items() if k in included_attrs}

        args = di
        type_ = "chat" if intent == "say" else "browser"
        tag = di.pop("tag") if "tag" in di else None

        return cls(
            index=index,
            intent=intent,
            args=args,
            type=type_,
            timestamp=timestamp,
            attrs=attrs,
            tag=tag,
        )

    def to_dict(
        self,
        include_timestamp=True,
        include_attrs=True,
        include_tag=True,
        include_index=True,
        drop_none_coords=False,
        format_timestamp_fn=None,
        ignore_args=None,
    ):
        """
        Convert the action to a dictionary, given specific options.

        Parameters
        ----------
        include_timestamp: bool
            Whether to include the timestamp in the output dictionary, as "timestamp"
        include_attrs: bool
            Whether to include the attributes in the output dictionary, as "attrs"
        include_tag: bool
            Whether to include the tag in the output dictionary, as "tag"
        include_index: bool
            Whether to include the index in the output dictionary, as "index"
        ignore_args: list
            A list of keys to ignore in the args dictionary, if None, then all keys are included
        format_timestamp_fn: callable
            A function to format the timestamp, if None, then the raw timestamp is used
        start_time: float
            The start time of the action, used to calculate the timestamp
        
        Returns
        -------
        dict
            A dictionary representation of the action.
        """
        if ignore_args is not None:
            args = {k: v for k, v in self.args.items() if k not in ignore_args}
        else:
            args = self.args

        out = {"intent": self.intent, **args}

        if include_tag and self.tag is not None:
            out["tag"] = self.tag

        if include_attrs and self.attrs is not None:
            out["attrs"] = self.attrs

        if include_timestamp:
            if format_timestamp_fn is not None:
                out["timestamp"] = format_timestamp_fn(self)["timestamp"]
            else:
                out["timestamp"] = self.timestamp

        if include_index:
            out["index"] = self.index

        if drop_none_coords:
            if "x" in out and out["x"] is None:
                del out["x"]
            if "y" in out and out["y"] is None:
                del out["y"]

        return out

    def to_str(self, **kwargs):
        """
        Converts the `Action` instance to a formatted string.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to pass to the `to_dict` method.

        Returns
        -------
        str
            A formatted string representation of the action.

        Notes
        -----

        This runs the `to_dict` method and then formats the output dictionary as a string, using
        `weblinx.utils.format.format_output_dictionary` with the intent as the "function" key.
        """
        di = self.to_dict(**kwargs)
        return format_output_dictionary(di, function_key="intent", return_as=str)

    def items(self):
        """
        Mimics `weblinx.Turn.items()` to retrieve dictionary items of the action.

        Returns
        -------
        ItemsView
            A view object that displays a list of a dictionary's key-value tuple pairs.
        
        Notes
        -----

        This method is aimed to mimic `weblinx.Turn.items()`
        """
        di = self.to_dict(
            include_timestamp=True,
            include_attrs=False,
            include_tag=False,
            include_index=False,
            drop_none_coords=True,
        )

        return di.items()
