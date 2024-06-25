from functools import partial
import weblinx.utils.format as wlf

def build_formatters_action_model() -> callable:
    """
    Builds and returns a dictionary of formatters for action model events.

    This function uses partial functions from the `weblinx.utils.format` module to create 
    formatters for various user actions, such as clicks, text inputs, changes, etc. These 
    formatters are then combined into a single formatter function for automatically formatting 
    intents based on user actions.

    Returns
    -------
    function
        A function that formats intents automatically using the defined formatters.
    
    Notes
    -----
    
    Slightly improved over original implementation from weblinx:
    https://github.com/McGill-NLP/weblinx/blob/7f151eaf819a9665b9b0b2232a99db6d4c4d2738/modeling/llama/processing.py#L23
    """
    format_click = partial(wlf.format_click, formatters=(wlf.format_uid,))
    format_text_input = partial(
        wlf.format_text_input,
        formatters=(
            partial(wlf.format_arg_item, name="text", max_length=200),
            wlf.format_uid,
        ),
    )
    format_change = partial(
        wlf.format_change,
        formatters=(
            partial(wlf.format_arg_item, name="value", max_length=200),
            wlf.format_uid,
        ),
    )
    format_copy = partial(wlf.format_copy, include_timestamp=False)
    format_submit = partial(wlf.format_submit, formatters=(wlf.format_uid,))
    format_load = partial(
        wlf.format_load,
        include_transition=False,
        include_timestamp=False,
        max_length=200,
    )
    format_hover = partial(wlf.format_hover, formatters=(wlf.format_uid,))
    format_paste = partial(wlf.format_paste, include_timestamp=False)
    format_scroll = partial(wlf.format_scroll, include_timestamp=False)
    format_say = partial(wlf.format_say, include_timestamp=False)
    format_tab = wlf.format_tab

    format_intent_auto = partial(
        wlf.format_intent_automatically,
        format_change=format_change,
        format_click=format_click,
        format_copy=format_copy,
        format_hover=format_hover,
        format_load=format_load,
        format_paste=format_paste,
        format_say=format_say,
        format_scroll=format_scroll,
        format_submit=format_submit,
        format_tab=format_tab,
        format_text_input=format_text_input,
    )

    return format_intent_auto


def build_formatters_dmr():
    """
    Builds and returns two dictionaries of formatters for DMR (Document Model Retrieval) events.

    This function creates formatters for both input and output events using partial functions 
    from the `weblinx.utils.format` module. For inputs, it formats elements, clicks, changes, 
    hovers, submits, and text inputs. For outputs, it formats elements, clicks, changes, loads, 
    scrolls, and text inputs.

    Returns
    -------
    tuple of functions
        A tuple containing two functions: one for formatting input intents and one for formatting 
        output intents.

    
    Examples
    -----

    ```python
    format_intent_input, format_intent_out = build_formatters_dmr()
    ```
    
    """
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
        format_tab=wlf.format_tab,
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

