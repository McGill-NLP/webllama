from collections import defaultdict
from copy import deepcopy
import random
from typing import Any, Dict, List
from functools import partial

import lxml.html
import weblinx as wl
import weblinx.utils.html as wh
import weblinx.utils.format as wlf
from weblinx.processing.prompt import (
    format_prev_turns,
    find_turns_with_instructor_chat,
    format_utterances,
)


def format_turn_for_input(
    replay,
    turn,
    format_intent,
    turn_sep=" ; ",
    num_prev_turns=5,
    num_utterances=5,
    return_str=True,
):
    """
    This function formats a turn for input to the model. It does so by combining the following:
    1. The first and last `num_utterances-1` utterances from the instructor
    2. The previous turns (up to `num_prev_turns` turns)

    If return_str is True, then the output is a string. Otherwise, it returns two strings: the utterance context and the previous turns.
    """
    prev_turns_text = format_prev_turns(
        replay=replay,
        turn=turn,
        format_intent=format_intent,
        turn_sep=turn_sep,
        num_prev_turns=num_prev_turns,
    )
    instructor_chat_turns = find_turns_with_instructor_chat(
        replay, turn, num_prev_turns=num_prev_turns
    )
    utterance_context = format_utterances(
        instructor_chat_turns, num_utterances=num_utterances
    )

    if not return_str:
        return utterance_context, prev_turns_text

    # Now, let's combine the text from the previous turns with the utterance context
    # and the current turn's utterance
    text = (
        f"Viewport(height={turn.viewport_height}, width={turn.viewport_width}) ---- "
        f"Instructor Utterances: {utterance_context} ---- "
        f"Previous Turns:{prev_turns_text}"
    )

    return text


def build_formatters():
    format_element_input = partial(
        wlf.format_element,
        include_text=False,
        include_attrs=("class", "title", "href", "aria-label", "d", "src"),
    )
    format_click_input = partial(
        wlf.format_click,
        formatters=(wlf.format_mouse_xy, format_element_input, wlf.format_timestamp),
    )
    format_change_input = partial(
        wlf.format_change,
        formatters=(
            partial(wlf.format_arg_item, name="value"),
            format_element_input,
            wlf.format_timestamp,
        ),
    )
    format_hover_input = partial(
        wlf.format_hover,
        formatters=(wlf.format_mouse_xy, format_element_input, wlf.format_timestamp),
    )

    format_submit_input = partial(
        wlf.format_submit, formatters=(format_element_input, wlf.format_timestamp)
    )

    format_text_input_input = partial(
        wlf.format_text_input,
        formatters=(
            partial(wlf.format_arg_item, name="text"),
            partial(format_element_input),
            wlf.format_timestamp,
        ),
    )

    format_intent_input = partial(
        wlf.format_intent_automatically,
        format_click=format_click_input,
        format_change=format_change_input,
        format_hover=format_hover_input,
        format_submit=format_submit_input,
        format_text_input=format_text_input_input,
        return_as=str,
    )

    # second, for the output (prediction text)
    format_element_out = partial(
        wlf.format_element,
        # Only want the tag
        include_text=False,
        include_attrs=False,
    )

    format_click_out = partial(wlf.format_click, formatters=(wlf.format_mouse_xy,))
    format_text_input_out = partial(
        wlf.format_text_input,
        formatters=(
            partial(wlf.format_arg_item, name="text", max_length=200),
            format_element_out,
            wlf.format_target_bbox,
        ),
    )
    format_change_out = partial(
        wlf.format_change,
        formatters=(
            partial(wlf.format_arg_item, name="value", max_length=200),
            format_element_out,
            wlf.format_target_bbox,
        ),
    )
    format_submit_out = partial(
        wlf.format_submit, formatters=(format_element_out, wlf.format_target_bbox)
    )
    format_load_out = partial(
        wlf.format_load,
        include_transition=False,
        include_timestamp=False,
        max_length=200,
    )
    format_scroll_out = partial(wlf.format_scroll, include_timestamp=False)

    format_say_out = partial(wlf.format_say, include_timestamp=False)

    format_intent_out = partial(
        wlf.format_intent_automatically,
        format_change=format_change_out,
        format_click=format_click_out,
        format_load=format_load_out,
        format_say=format_say_out,
        format_scroll=format_scroll_out,
        format_submit=format_submit_out,
        format_text_input=format_text_input_out,
    )

    return format_intent_input, format_intent_out


def turn_has_valid_uid(turn, paths, uid_key="data-webtasks-id"):
    """
    Given a turn an lxml tree, return True if the turn's uid is in the tree.
    """
    uids = [p.attrib[uid_key] for p in paths]
    if turn.element is None or uid_key not in turn.element["attributes"]:
        return False

    if turn.element["attributes"][uid_key] not in uids:
        return False

    return True


def format_attrs(attrs):
    return " ".join([f"{k!s}={v!r}" for k, v in attrs.items()])


def shorten(s, max_length=100, side="center", ellipsis="..."):
    if max_length is None:
        return s

    if len(s) <= max_length:
        return s

    max_length = max_length - len(ellipsis)

    if side == "right":
        s = s[:max_length] + ellipsis
    elif side == "left":
        s = ellipsis + s[-max_length:]
    elif side == "center":
        s = s[: max_length // 2] + ellipsis + s[-max_length // 2 :]
    else:
        raise ValueError(f"Invalid side: {side}")

    return s


def format_children(parent, depth=1):
    """
    Use the concise parentheses notation to format the children of an element.
    For example, for depth 1, we only have: (child1 child2 child3)
    For depth 2, we have: (child1 (grandchild1 grandchild2) child2 child3)
    """
    children = parent.getchildren()
    if len(children) == 0:
        return ""

    if depth == 1:
        return " ".join([c.tag for c in children])

    out_str = ""
    for c in children:
        out_str += f"{c.tag}"
        children_str = format_children(c, depth=depth - 1)
        if children_str != "":
            out_str += f" ( {children_str} )"
        out_str += " "

    return out_str.strip()


def represent_element_as_dict(
    element,
    bbox,
    root_tree,
    max_text_length=200,
    max_attr_length=100,
    max_child_depth=2,
):
    """
    Format an lxml element into a dictionary of strings. The keys are:
    - tag: the tag name of the element
    - xpath: the xpath of the element
    - text: the text of the element, truncated to `max_text_length`
    - bbox: the bounding box of the element
    - attributes: the attributes of the element, truncated to `max_attr_length`
    - children: the children of the element, truncated to `max_attr_length`
    """
    # Get the tag name
    tag = element.tag
    xpath = root_tree.getpath(element)
    children = element.getchildren()
    text = element.text if element.text is not None else ""

    # Shorten the text and attributes
    text = shorten(text, max_text_length)
    attrs = {k: shorten(v, max_attr_length) for k, v in element.attrib.items()}

    # Sort the attributes by length
    attrs = dict(sorted(attrs.items(), key=lambda x: len(x[1])))

    # Truncate the children
    children = children[:max_child_depth]

    # Format the children
    children_str = " ".join([c.tag for c in children if isinstance(c.tag, str)])
    children_str = shorten(children_str, max_attr_length)

    # Format the attributes
    attrs_str = format_attrs(attrs)

    # Format the bounding box
    bbox_str = " ".join(
        [f"{k}={round(bbox[k], 1)}" for k in ["x", "y", "width", "height"]]
    )

    # format as a dict
    element_dict = {
        "tag": tag,
        "xpath": xpath,
        "text": text,
        "bbox": bbox_str,
        "attributes": attrs_str,
        "children": children_str,
    }

    return element_dict


def convert_elem_dict_to_str_legacy(elem_dict: dict):
    """
    Convert an element dictionary to a string.
    """
    elem_dict = deepcopy(elem_dict)

    element_str = f"[[tag]] {elem_dict.pop('tag')}\n"
    element_str += f"[[xpath]] {elem_dict.pop('xpath')}\n"
    element_str += f"[[text]] {elem_dict.pop('text')}\n"
    element_str += f"[[bbox]] {elem_dict.pop('bbox')}\n"
    element_str += f"[[attributes]] {elem_dict.pop('attributes')}\n"
    element_str += f"[[children]] {elem_dict.pop('children')}"

    # for other keys, we just add them to the end

    for k, v in elem_dict.items():
        element_str += f"\n[[{k}]] {v}"

    return element_str


def build_records_for_single_turn(
    turn, replay, format_intent_input, uid_key, max_neg=None, only_allow_valid_uid=True
) -> List[dict]:
    """
    This function will build a list of dictionaries, each of which is a record
    for a single turn. Each record has the following keys:
        - query: the dialogue history, used as a query for training the model
        - doc: concise representation of HTML element used as doc for training
        - label: either 0 or 1, indicating whether the document is the target element
        - uid: the unique identifier for an element, must be in the element attributes
        - turn_index: the index of the turn in the replay
        - demo_name: the name of the demonstration

    If `only_allow_valid_uid` is True, then only turns that have a valid uid
    will be included in the output. Otherwise, all turns will be included.
    """
    bboxes_filt = wh.filter_bboxes(
        turn.bboxes,
        viewport_height=turn.viewport_height,
        viewport_width=turn.viewport_width,
    )
    root = lxml.html.fromstring(turn.html)
    root_tree = root.getroottree()
    elements = root.xpath(f"//*[@{uid_key}]")
    elements_filt = [p for p in elements if p.attrib[uid_key] in bboxes_filt]

    has_valid_uid = turn_has_valid_uid(turn, paths=elements, uid_key=uid_key)
    if only_allow_valid_uid and not has_valid_uid:
        return []

    # Now, we can format each of the elements in paths_filt into string
    # and use them as negative samples
    query = format_turn_for_input(replay, turn, format_intent=format_intent_input)
    target_uid = turn.element["attributes"][uid_key] if has_valid_uid else -1

    records_positive = []
    records_negative = []

    for elem in elements_filt:
        bbox = turn.bboxes[elem.attrib[uid_key]]
        elem_dict = represent_element_as_dict(elem, bbox, root_tree)
        elem_str = convert_elem_dict_to_str_legacy(elem_dict)

        record = {
            "query": query,
            "doc": elem_str,
            "uid": elem.attrib[uid_key],
            "demo_name": turn.demo_name,
            "turn_index": turn.index,
            "elem_dict": elem_dict,
        }

        if elem.attrib[uid_key] == target_uid:
            record["label"] = 1
            records_positive.append(record)
        else:
            record["label"] = 0
            records_negative.append(record)

    if max_neg is not None and 0 < max_neg < len(records_negative):
        records_negative = random.sample(records_negative, max_neg)

    return records_positive + records_negative


def build_records_for_single_demo(
    demo,
    format_intent_input,
    max_neg_per_turn=None,
    random_state=None,
    uid_key="data-webtasks-id",
    only_allow_valid_uid=True,
    group_by_turn=False,
) -> List[dict]:
    """
    This runs `build_records_for_single_turn` for each turn in the demonstration.
    First, the demonstration is converted into a replay, and then we filter the
    turns to only those that have HTML and bounding boxes, and that are of the
    following intents:
        - click
        - change
        - textInput
        - scroll
        - load
        - submit

    Any turn that does not have a valid uid is discarded.
    """
    if random_state is not None:
        random.seed(random_state)

    replay = wl.Replay.from_demonstration(demo)
    turns = replay.filter_by_intents(
        "click", "change", "textInput", "scroll", "load", "submit"
    )
    turns = wl.filter_turns(turns, lambda t: t.has_html() and t.has_bboxes())

    records_for_demo = []
    for turn in turns:
        recs = build_records_for_single_turn(
            turn=turn,
            replay=replay,
            format_intent_input=format_intent_input,
            uid_key=uid_key,
            max_neg=max_neg_per_turn,
            only_allow_valid_uid=only_allow_valid_uid,
        )
        if group_by_turn:
            records_for_demo.append(recs)
        else:
            records_for_demo.extend(recs)

    return records_for_demo
