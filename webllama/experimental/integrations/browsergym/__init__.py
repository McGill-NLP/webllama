from copy import deepcopy
import random
import lxml.html


def postprocess_for_browsergym(action, uid_map=None):
    # if uid is a int, we need to convert it to a string
    uid_map = {} if uid_map is None else uid_map

    if "uid" in action:
        action["uid"] = str(action["uid"])
        if action["uid"] in uid_map:
            action["uid"] = uid_map[action["uid"]]

    action = deepcopy(action)
    if action["intent"] == "scroll":
        if not "x" in action:
            action["x"] = 0
        if not "y" in action:
            action["y"] = 0

    return action


def generate_uuid(old_attr_name):
    # We do not use old_attr_name here, but it is required by the signature of the function.
    def replace_char(c):
        r = random.randint(0, 15)
        v = r if c == "x" else (r & 0x3 | 0x8)
        return format(v, "x")

    uuid_template = "xxxxxxxx-xxxx-4xxx"
    return "".join(replace_char(c) if c in "xy" else c for c in uuid_template)


def reverse_dict(mapping):
    return {v: k for k, v in mapping.items()}

def replace_bid_with_wl_uid(
    dom_str,
    new_attr_name="data-webtasks-id",
    old_attr_name="bid",
    generate_fn=generate_uuid,
    return_mapping=False,
):
    """
    Replaces the bid attributes in the dom string with a new attribute name and a new unique id.

    generate_fn must be a function that takes the old attribute name and returns a new unique id.
    """
    html_parsed = lxml.html.fromstring(dom_str)

    new_attr_mapping = {
        str(elem.get(old_attr_name)): generate_fn(old_attr_name)
        for elem in html_parsed.xpath(f"//*[@{old_attr_name}]")
        if elem.get(old_attr_name) is not None
    }

    # remap the attributes from bid="key" to data-webtasks-id="value"
    for elem in html_parsed.xpath("//*[@bid]"):
        elem.set(new_attr_name, new_attr_mapping[elem.get(old_attr_name)])
        elem.attrib.pop(old_attr_name)

    html_processed_str = lxml.html.tostring(html_parsed).decode("utf-8")

    if return_mapping:
        return html_processed_str, new_attr_mapping
    else:
        return html_processed_str