import time
from typing import Dict, List, TYPE_CHECKING
import typing

import lxml.html
import weblinx.utils.html as wh
from weblinx.processing.prompt import (
    find_turns_with_instructor_chat,
    format_utterances,
    get_speaker,
    format_prev_turns,
)
from weblinx.processing.outputs import (
    parse_predicted_output_string,
    sanitize_args,
)
from weblinx.processing.truncation import (
    convert_elem_dict_to_str_dmr,
)

from .classes import Action, BBox, State

if TYPE_CHECKING:
    from transformers import AutoTokenizer

# Custom types
UID = typing.NewType("UID", str)
DEFAULT_FINAL_USER_MSG = "Please select the best action using the correct format, do not provide any other information or explanation."


def _shorten(s, max_length=100, side="center", ellipsis="..."):
    """
    Shortens a string to a specified maximum length, adding an ellipsis if necessary.

    Parameters
    ----------
    s : str
        The string to be shortened.
    max_length : int, optional
        The maximum length of the shortened string, including the ellipsis. Default is 100.
    side : str, optional
        The side from which to shorten the string. Options are "center", "left", and "right". Default is "center".
    ellipsis : str, optional
        The string to use as an ellipsis. Default is "...".

    Returns
    -------
    str
        The shortened string.
    """
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


def _format_attrs(attrs):
    """
    Formats a dictionary of attributes into a string.

    Parameters
    ----------
    attrs : dict
        The dictionary of attributes to format.

    Returns
    -------
    str
        The formatted attributes as a string.
    """
    return " ".join([f"{k!s}={v!r}" for k, v in attrs.items()])


def _extract_output_str_from_pipeline_output(output):
    """
    Extracts the output string from a pipeline output.

    Parameters
    ----------
    output : str or dict or list
        The output from the pipeline, which can be a string, a dictionary, or a list.

    Returns
    -------
    str
        The extracted output string.
    """
    if isinstance(output, str):
        output_str = output
    elif isinstance(output, dict):
        if "generated_text" not in output:
            raise ValueError("Output dictionary does not have 'generated_text' key.")
        output_str = output["generated_text"]

    elif isinstance(output, list):
        if len(output) == 0:
            raise ValueError("Output list is empty, cannot be processed.")
        if len(output) > 1:
            raise ValueError(
                "Output list has more than one element, cannot be processed."
            )
        o = output[0]

        if isinstance(o, str):
            output_str = o
        elif isinstance(o, dict):
            if "generated_text" not in o:
                raise ValueError(
                    "Output dictionary does not have 'generated_text' key."
                )
            output_str = o["generated_text"]
        else:
            raise ValueError(
                f"Output list has element of type {type(o)}, cannot be processed."
            )

    return output_str


def represent_element_as_dict(
    element,
    bbox,
    root_tree,
    max_text_length=200,
    max_attr_length=100,
    max_child_depth=2,
    return_attrs=True,
):
    """
    Format an lxml element into a dictionary of strings.

    Parameters
    ----------
    element : lxml.html.Element
        The element to format.
    bbox : dict
        The bounding box of the element.
    root_tree : lxml.html.ElementTree
        The root tree of the document.
    max_text_length : int, optional
        The maximum length of the text. Default is 200.
    max_attr_length : int, optional
        The maximum length of the attributes. Default is 100.
    max_child_depth : int, optional
        The maximum depth of children to include. Default is 2.
    return_attrs : bool, optional
        Whether to return the attributes. Default is True.

    Returns
    -------
    dict
        A dictionary representation of the element.

    Notes
    -----
    We expect the following keys in the output dictionary:
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
    text = _shorten(text, max_text_length)
    attrs = {k: _shorten(v, max_attr_length) for k, v in element.attrib.items()}

    # Sort the attributes by length
    attrs = dict(sorted(attrs.items(), key=lambda x: len(x[1])))

    # Truncate the children
    children = children[:max_child_depth]

    # Format the children
    children_str = " ".join([c.tag for c in children if isinstance(c.tag, str)])
    children_str = _shorten(children_str, max_attr_length)

    # Format the attributes
    attrs_str = _format_attrs(attrs)

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

    if return_attrs:
        return element_dict, attrs
    else:
        return element_dict


def calculate_remaining_tokens_before_candidates(
    html: str,
    utterance_context: str,
    prev_actions_as_chat: List[Dict[str, str]],
    tokenizer: "AutoTokenizer",
    max_html_tokens: int,
    max_utterance_tokens: int,
    max_prev_turns_tokens: int,
    tokenizer_chat: "AutoTokenizer" = None,
):
    """
    Calculates the remaining tokens before candidates can be processed.

    Parameters
    ----------
    html : str
        The HTML content.
    utterance_context : str
        The context of the current utterance.
    prev_actions_as_chat : list of dict
        A list of previous actions formatted as chat.
    tokenizer : AutoTokenizer
        The tokenizer for the HTML content.
    max_html_tokens : int
        The maximum number of HTML tokens.
    max_utterance_tokens : int
        The maximum number of utterance tokens.
    max_prev_turns_tokens : int
        The maximum number of previous turn tokens.
    tokenizer_chat : AutoTokenizer, optional
        The tokenizer for the chat content. Default is None.

    Returns
    -------
    int
        The remaining tokens before candidates.
    """
    if tokenizer_chat is None:
        tokenizer_chat = tokenizer

    num_prev_turns_tokens = len(
        tokenizer_chat.apply_chat_template(
            [{"role": "system", "content": ""}, *prev_actions_as_chat], tokenize=True
        )
    )
    # Add the unused length to the candidates
    num_html_tokens = len(tokenizer.tokenize(html))
    num_utter_tokens = len(tokenizer.tokenize(utterance_context))
    remain_html_tokens = max_html_tokens - num_html_tokens
    remain_utter_tokens = max_utterance_tokens - num_utter_tokens
    remain_prev_turns_tokens = max_prev_turns_tokens - num_prev_turns_tokens
    remain_tokens = remain_html_tokens + remain_utter_tokens + remain_prev_turns_tokens
    # Add the unused length to the max_candidates_tokens
    remain_tokens

    return remain_tokens


def convert_prev_actions_to_chat(
    prev_actions_text_list,
    final_user_message=DEFAULT_FINAL_USER_MSG,
    add_dummy_user_first="auto",
):
    """
    Converts previous actions to a chat format compatible with huggingface
    (e.g. the `tokenizer.apply_chat_template` method) and openai.

    Parameters
    ----------
    prev_actions_text_list : list of str
        A list of previous actions as text.
    final_user_message : str, optional
        The final user message to include. Default is DEFAULT_FINAL_USER_MSG.
    add_dummy_user_first : str or bool, optional
        Whether to add a dummy user message first. Default is 'auto'.

    Returns
    -------
    list of dict
        A list of previous actions formatted as chat.
    """
    prev_turns_merged = []

    # Merge turns from the same role
    for i, action_text in enumerate(prev_actions_text_list):
        role = get_speaker(
            action_text,
            instructor_name="user",
            navigator_name="assistant",
            default_name="unknown",
        )

        if i > 0 and prev_turns_merged[-1]["role"] == role:
            prev_turns_merged[-1]["content"] += " " + action_text
        else:
            prev_turns_merged.append({"role": role, "content": action_text})

    if len(prev_turns_merged) > 0 and prev_turns_merged[-1]["role"] == "user":
        prev_turns_merged[-1]["content"] += " " + final_user_message
    else:
        prev_turns_merged.append({"role": "user", "content": final_user_message})

    if add_dummy_user_first == "auto":
        # only add dummy user if the user has not provided any input at the beginning
        add_dummy_user_first = prev_turns_merged[0]["role"] == "assistant"

    if add_dummy_user_first is True:
        # This is needed in case the user has not provided any input,
        # since the tokenizet apply_chat_template will not work if there's no user/assistant
        # alternating
        prev_turns_merged.insert(0, {"role": "user", "content": ""})

    return prev_turns_merged


def prepare_query_for_dmr(
    action_history: List[Action],
    state: State,
    format_intent,
    turn_sep=" ; ",
    num_prev_turns=5,
    num_utterances=5,
    return_str=True,
):
    """
    Formats a turn for query input to the DMR model.

    Parameters
    ----------
    action_history : list of Action
        The history of actions.
    state : State
        The current state.
    format_intent : callable
        The function to format intents.
    turn_sep : str, optional
        The separator for turns. Default is " ; ".
    num_prev_turns : int, optional
        The number of previous turns to include. Default is 5.
    num_utterances : int, optional
        The number of utterances to include. Default is 5.
    return_str : bool, optional
        Whether to return the result as a string, or as a 2-tuple of strings, one for the utterance context
        and one for the previous turns. Default is True.

    Returns
    -------
    str or tuple of str
        The formatted query, either as a single string or as a tuple of strings.

    Notes
    -----
    To format a turn for query input to the DMR model, we combine the following:

    1. The first and last `num_utterances-1` utterances from the instructor
    2. The previous turns (up to `num_prev_turns` turns)
    """
    prev_turns_text = format_prev_turns(
        replay=action_history,
        turn=state,
        format_intent=format_intent,
        turn_sep=turn_sep,
        num_prev_turns=num_prev_turns,
    )
    instructor_chat_turns = find_turns_with_instructor_chat(
        action_history, state, num_prev_turns=num_prev_turns
    )
    utterance_context = format_utterances(
        instructor_chat_turns, num_utterances=num_utterances
    )

    if not return_str:
        return utterance_context, prev_turns_text

    # Now, let's combine the text from the previous turns with the utterance context
    # and the current turn's utterance
    text = (
        f"Viewport(height={state.viewport_height}, width={state.viewport_width}) ---- "
        f"Instructor Utterances: {utterance_context} ---- "
        f"Previous Turns:{prev_turns_text}"
    )

    return text


def prepare_dmr_elements(
    state: State, uid_key: str = "data-webtasks-id", bboxes: BBox = None
) -> List[dict]:
    """
    Builds a list of dictionaries representing DMR elements.

    Parameters
    ----------
    state : State
        The current state.
    uid_key : str, optional
        The key for the unique identifier. Default is "data-webtasks-id".
    bboxes : BBox, optional
        The bounding boxes of elements. Default is None.

    Returns
    -------
    list of dict
        A list of dictionaries (i.e. record) representing DMR elements.

    Notes
    -----

    Each record in the output has the following keys:
    - query: the dialogue history, used as a query for training the model
    - doc: concise representation of HTML element used as doc for training
    - label: either 0 or 1, indicating whether the document is the target element
    - uid: the unique identifier for an element, must be in the element attributes
    - turn_index: the index of the turn in the replay
    - demo_name: the name of the demonstration
    """
    if bboxes is None:
        bboxes = state.bboxes

    bboxes_filt = wh.filter_bboxes(
        bboxes,
        viewport_height=state.viewport_height,
        viewport_width=state.viewport_width,
    )
    root = lxml.html.fromstring(state.html)
    root_tree = root.getroottree()
    elements = root.xpath(f"//*[@{uid_key}]")
    elements_filt = [p for p in elements if p.attrib[uid_key] in bboxes_filt]
    candidates = []

    for elem in elements_filt:
        bbox = bboxes[elem.attrib[uid_key]]
        elem_dict, attrs = represent_element_as_dict(elem, bbox, root_tree)
        elem_str = convert_elem_dict_to_str_dmr(elem_dict)

        record = {
            "doc": elem_str,
            "uid": elem.attrib[uid_key],
            "elem_dict": elem_dict,
            "attrs": attrs,
        }
        candidates.append(record)

    return candidates


def cos_sim(a, b):
    """
    Computes the cosine similarity between two matrices.

    Parameters
    ----------
    a : torch.Tensor
        The first matrix.
    b : torch.Tensor
        The second matrix.

    Returns
    -------
    torch.Tensor
        A matrix with cosine similarities between all pairs of rows in a and b.

    Notes
    -----
    Taken from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
    Use of this function is subject to the following license: https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE
    """
    try:
        import torch
    except:
        raise ImportError("This function requires PyTorch to be installed.")

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def compute_dmr_scores(dmr, query, cands, batch_size=16):
    """
    Computes the DMR scores for candidate elements.

    Parameters
    ----------
    dmr : callable
        The DMR model.
    query : str
        The query string, returned by `prepare_query_for_dmr`.
    cands : list of dict
        The candidate elements, returned by `prepare_dmr_elements`.
    batch_size : int, optional
        The batch size for encoding. Default is 16.

    Returns
    -------
    list of float
        The DMR scores for the candidate elements.

    Examples
    --------

    ```python
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    dmr = SentenceTransformer(dmr_name, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(act_model)
    proc = wa.processing.WebTurnProcessor(tokenizer=tokenizer)

    query_dmr = proc.prepare_dmr_query(action_history, state)
    elems = proc.prepare_dmr_elements(state=state)
    scores = wa.functions.compute_dmr_scores(dmr, query_dmr, elems)
    ```
    """
    cands_docs = [cand["doc"] for cand in cands]
    encoded = dmr.encode(
        [query] + cands_docs, batch_size=batch_size, show_progress_bar=False
    )
    query_vector, doc_vectors = encoded[0], encoded[1:]
    scores = cos_sim(query_vector, doc_vectors).cpu().squeeze().tolist()

    return scores


def get_top_dmr_candidates(candidate_elements, scores, top_k):
    """
    Gets the top DMR candidates based on scores.

    Parameters
    ----------
    candidate_elements : list of dict
        The candidate elements.
    scores : list of float
        The scores for the candidate elements.
    top_k : int
        The number of top candidates to return.

    Returns
    -------
    tuple of (list of dict, list of str)
        The top candidates and their unique identifiers.
    """
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_cands_indices = indices[:top_k]
    top_cands = [candidate_elements[i] for i in top_cands_indices]
    cands_uids = [cand["uid"] for cand in top_cands]

    return top_cands, cands_uids


def infer_tag_and_attrs(
    args, elems, accepted=("class", "title", "href", "aria-label", "d", "src")
):
    """
    Infers the tag and attributes for an element based on its arguments. This is an internal function
    used by the `process_action_model_output`.

    Parameters
    ----------
    args : dict
        The arguments for the element.
    elems : list of dict
        The elements to infer from.
    accepted : tuple of str, optional
        The accepted attributes. Default is ("class", "title", "href", "aria-label", "d", "src").

    Returns
    -------
    tuple of (str, dict)
        The inferred tag and attributes.
    """
    if "uid" in args:
        uids_to_elems = {elem["uid"]: elem for elem in elems}
        uid = args["uid"]
        if uid in uids_to_elems:
            elem_dict = uids_to_elems[uid]["elem_dict"]
            # need to take elem_dict's attributes_dict and bbox from state.bboxes
            attrs = {
                k: v for k, v in uids_to_elems[uid]["attrs"].items() if k in accepted
            }
            tag = elem_dict["tag"]
        else:
            attrs, tag = {}, None

    else:
        attrs, tag = {}, None

    return tag, attrs


def create_action_history_from_replay(
    replay, format_intent_input_dmr, format_intent_am, stop_at_idx=None
):
    """
    Creates an action history from a replay of actions. The replay must be from
    WebLINX, or a strictly compatible format.

    Parameters
    ----------
    replay : list of Action
        The replay of actions.
    format_intent_input_dmr : callable
        The function to format DMR input intents.
    format_intent_am : callable
        The function to format action model intents.
    stop_at_idx : int, optional
        The index to stop at in the replay. Default is None.

    Returns
    -------
    list of Action
        The created action history.
    """
    if stop_at_idx is None:
        stop_at_idx = len(replay)
    else:
        stop_at_idx = min(stop_at_idx, len(replay))

    idx = stop_at_idx
    action_history: List[Action] = []
    for t in replay[:idx]:
        dmr_input = format_intent_input_dmr(t, return_as=dict)
        dictionary = format_intent_am(t, return_as=dict)
        dictionary["tag"] = dmr_input.get("tag", None)
        dictionary["x"] = dmr_input.get("x", None)
        dictionary["y"] = dmr_input.get("y", None)
        dictionary["attrs"] = dmr_input.get("attrs", None)
        dictionary["timestamp"] = t.timestamp
        dictionary["index"] = t.index

        action_history.append(Action.from_dict(dictionary))

    return action_history


def process_action_model_output(output, index, elems, start_time):
    """
    Processes the output of an action model to create a predicted action.

    Parameters
    ----------
    output : str or dict or list
        The output from the action model.
    index : int
        The index of the action.
    elems : list of dict
        The elements to process.
    start_time : float
        The start time of the action.

    Returns
    -------
    dict
        The predicted action.
    """
    output_str = _extract_output_str_from_pipeline_output(output)
    intent, args = parse_predicted_output_string(output_str)
    args = sanitize_args(args)
    tag, attrs = infer_tag_and_attrs(args=args, elems=elems)
    pred_action = {
        "index": index,
        "intent": intent,
        "timestamp": time.time() - start_time,
        "tag": tag,
        "attrs": attrs,
    }
    # only take args element that is not in pred_action
    pred_action.update({k: v for k, v in args.items() if k not in pred_action})

    return pred_action
