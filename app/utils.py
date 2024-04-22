#!/usr/bin/env python3

from datetime import datetime
import os
import json
from pathlib import Path
import sys
import shutil
import time
import traceback

import pandas as pd
import streamlit as st

import json
from PIL import Image, ImageDraw


CACHE_TTL = 60 * 60 * 24 * 14

"""
Streamlit app utilities
"""


@st.cache_data(ttl=CACHE_TTL)
def load_json(basedir, name):
    if not os.path.exists(f"{basedir}/{name}.json"):
        return None

    with open(f"{basedir}/{name}.json", "r") as f:
        j = json.load(f)

    return j


def load_json_no_cache(basedir, name):
    if not os.path.exists(f"{basedir}/{name}.json"):
        return None

    with open(f"{basedir}/{name}.json", "r") as f:
        j = json.load(f)

    return j


def save_json(basedir, name, data):
    with open(f"{basedir}/{name}.json", "w") as f:
        json.dump(data, f, indent=4)


@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img


@st.cache_resource
def load_page(page_path):
    return open(page_path, "rb")


def shorten(s):
    # shorten to 100 characters
    if len(s) > 100:
        s = s[:100] + "..."

    return s


@st.cache_data
def parse_arguments(action):
    s = []
    event_type = action["intent"]
    args = action["arguments"]

    if event_type == "textInput":
        txt = args["text"]

        txt = txt.strip()

        # escape markdown characters
        txt = txt.replace("_", "\\_")
        txt = txt.replace("*", "\\*")
        txt = txt.replace("`", "\\`")
        txt = txt.replace("$", "\\$")

        txt = shorten(txt)

        s.append(f'"{txt}"')
    elif event_type == "change":
        s.append(f'{args["value"]}')
    elif event_type == "load":
        url = args["properties"].get("url") or args.get("url")
        short_url = shorten(url)
        s.append(f'"[{short_url}]({url})"')

        if args["properties"].get("transitionType"):
            s.append(f'*{args["properties"]["transitionType"]}*')
            s.append(f'*{" ".join(args["properties"]["transitionQualifiers"])}*')
    elif event_type == "scroll":
        s.append(f'{args["scrollX"]}, {args["scrollY"]}')
    elif event_type == "say":
        s.append(f'"{args["text"]}"')
    elif event_type == "copy":
        selected = shorten(args["selected"])
        s.append(f'"{selected}"')
    elif event_type == "paste":
        pasted = shorten(args["pasted"])
        s.append(f'"{pasted}"')
    elif event_type == "tabcreate":
        s.append(f'{args["properties"]["tabId"]}')
    elif event_type == "tabremove":
        s.append(f'{args["properties"]["tabId"]}')
    elif event_type == "tabswitch":
        s.append(
            f'{args["properties"]["tabIdOrigin"]} -> {args["properties"]["tabId"]}'
        )

    if args.get("element"):
        
        if event_type == 'click':
            x = round(args['metadata']['mouseX'], 1)
            y = round(args['metadata']['mouseY'], 1)
            uid = args.get('element', {}).get('attributes', {}).get("data-webtasks-id")
            s.append(f"*x =* {x}, *y =* {y}, *uid =* {uid}")
        else:
            top = round(args["element"]["bbox"]["top"], 1)
            left = round(args["element"]["bbox"]["left"], 1)
            right = round(args["element"]["bbox"]["right"], 1)
            bottom = round(args["element"]["bbox"]["bottom"], 1)
            
            s.append(f"*top =* {top}, *left =* {left}, *right =* {right}, *bottom =* {bottom}")

    return ", ".join(s)


@st.cache_resource(max_entries=50_000, ttl=CACHE_TTL)
def create_visualization(_img, event_type, bbox, x, y, screenshot_path):
    # screenshot_path is not used, but we need it for caching since we can't cache
    # PIL images (hence the leading underscore in the variable name to indicate
    # that it's not hashed)
    _img = _img.convert("RGBA")
    draw = ImageDraw.Draw(_img)

    # draw a bounding box around the element
    color = {
        "click": "red",
        "hover": "orange",
        "textInput": "blue",
        "change": "green",
        "submit": "purple",
    }[event_type]

    left = bbox["left"]
    top = bbox["top"]
    w = bbox["width"]
    h = bbox["height"]
    draw.rectangle((left, top, left + w, top + h), outline=color, width=2)

    if event_type in ["click", "hover"]:
        r = 15
        for i in range(1, 5):
            rx = r * i
            draw.ellipse((x - rx, y - rx, x + rx, y + rx), outline=color, width=3)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return _img


@st.cache_data(max_entries=50_000, ttl=CACHE_TTL)
def get_screenshot_minimal(screenshot_path, event_type, bbox, x, y, new_width=None, overlay=True):
    img = load_image(screenshot_path)
    # vis = None

    if event_type in ["click", "textInput", "change", "hover", "submit"] and overlay:
        img = create_visualization(img, event_type, bbox, x, y, screenshot_path)

    if new_width is not None:
        # Resize to 800px wide
        w, h = img.size
        new_w = new_width
        new_h = int(new_w * h / w)
        img = img.resize((new_w, new_h))
        print(f"Resized '{screenshot_path}' to", new_w, new_h)

    return img


def get_event_info(d):
    event_type = d["action"]["intent"]

    try:
        bbox = d["action"]["arguments"]["element"]["bbox"]
    except KeyError:
        bbox = None

    try:
        x = d["action"]["arguments"]["properties"]["x"]
        y = d["action"]["arguments"]["properties"]["y"]
    except KeyError:
        x = None
        y = None

    return event_type, bbox, x, y


def get_screenshot(d, basedir, new_width=None, overlay=True):
    screenshot_filename = d["state"]["screenshot"]

    if not screenshot_filename:
        return None

    event_type, bbox, x, y = get_event_info(d)
    screenshot_path = f"{basedir}/screenshots/{screenshot_filename}"

    return get_screenshot_minimal(
        screenshot_path, event_type, bbox, x, y, new_width=new_width, overlay=overlay
    )


def text_bubble(text, color):
    text = text.replace("\n", "<br>").replace("\t", "&nbsp;" * 8)
    return f'<div style="background-color:{color}; padding: 8px; margin: 6px; border-radius:10px; display:inline-block;">{text}</div>'


def gather_chat_history(data, example_index):
    chat = []
    for i, d in enumerate(data):
        if d["type"] == "chat":
            if i >= example_index:
                break
            chat.append(d)

    # # leave out just 5 last messages
    # if len(chat) > 5:
    #     chat = chat[-5:]

    return reversed(chat)


def format_chat_message(d):
    if d["speaker"] == "instructor":
        return text_bubble("🧑 " + d["utterance"], "rgba(63, 111, 255, 0.35)")
    else:
        return text_bubble("🤖 " + d["utterance"], "rgba(185,185,185,0.35)")


def find_screenshot(data, example_index, basedir, overlay=True):
    # keep looking at previous screenshots until we find one
    # if there is none, return None

    for i in range(example_index, -1, -1):
        d = data[i]
        if d["type"] == "chat":
            continue

        screenshot = get_screenshot(d, basedir, overlay=overlay)
        if screenshot:
            return screenshot

    return None


def create_visualization_2(_img, bbox, color, width, x, y):
    _img = _img.convert("RGBA")
    draw = ImageDraw.Draw(_img)

    if bbox:
        left = bbox["left"]
        top = bbox["top"]
        w = bbox["width"]
        h = bbox["height"]
        draw.rectangle((left, top, left + w, top + h), outline=color, width=width)

    if x and y:
        r = 8
        for i in range(1, 4):
            rx = r * i
            draw.ellipse((x - rx, y - rx, x + rx, y + rx), outline=color, width=2)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return _img


def rescale_bbox(bbox, scaling_factor):
    return {
        k: bbox[k] * scaling_factor
        for k in ["top", "left", "width", "height", "right", "bottom"]
        if k in bbox
    }


def show_overlay(
    _img,
    pred,
    ref,
    turn_args,
    turn_metadata,
    scale_pred=True,
    show=("pred_coords", "ref", "pred_elem"),
):
    scaling_factor = turn_metadata.get("zoomLevel", 1.0)

    if "pred_elem" in show:
        # First, draw red box around predicted element
        if pred.get("element") and pred["element"].get("bbox"):
            # rescale the bbox by scaling_factor
            bbox = rescale_bbox(pred["element"]["bbox"], scaling_factor)
            _img = create_visualization_2(
                _img, bbox, color="red", width=9, x=None, y=None
            )

    if "ref" in show:
        # Finally, draw a blue box around the reference element (if it exists)
        if ref.get("element") and ref["element"].get("bbox"):
            # rescale the bbox
            bbox = rescale_bbox(ref["element"]["bbox"], scaling_factor)
            x = turn_args.get("properties", {}).get("x")
            y = turn_args.get("properties", {}).get("y")
            _img = create_visualization_2(_img, bbox, color="blue", width=6, x=x, y=y)

    if "pred_coords" in show:
        # Second draw a green box and x/y coordinate based on predicted coordinates
        # The predicted coordinates are the raw output of the model,
        # Whereas the predicted element is the inferred element from the predicted coordinates
        if pred["args"].get("x") and pred["args"].get("y"):
            x = pred["args"]["x"]
            y = pred["args"]["y"]

            if scale_pred:
                x = x * scaling_factor
                y = y * scaling_factor
        else:
            x = None
            y = None

        # If the predicted element is a bounding box, draw a green box around it
        if all(c in pred["args"] for c in ["top", "left", "right", "bottom"]):
            bbox = {
                "top": pred["args"]["top"],
                "left": pred["args"]["left"],
                "width": (pred["args"]["right"] - pred["args"]["left"]),
                "height": (pred["args"]["bottom"] - pred["args"]["top"]),
                "right": pred["args"]["right"],
                "bottom": pred["args"]["bottom"],
            }

            if scale_pred:
                bbox = rescale_bbox(bbox, scaling_factor)
        else:
            # Otherwise, do nothing
            bbox = None

        _img = create_visualization_2(_img, bbox=bbox, color="green", width=3, x=x, y=y)

    return _img



def get_zoom_level(d):
    """
    Get the zoom level of the page
    """

    # If it's type chat, we just set the zoom level to 1 and ignore
    if d["type"] == "chat":
        return 100

    # the zoom level is in the state of the turn
    # d is the turn
    # the zoom level is in d['state']['zoom']
    # if it is not present, return 100
    option1 = (
        d.get("action", {})
        .get("arguments", {})
        .get("properties", {})
        .get("zoomLevel")
    )
    option2 = (
        d.get("action", {})
        .get("arguments", {})
        .get('metadata', {})
        .get("zoomLevel")
    )

    if option1 is not None:
        return option1
    elif option2 is not None:
        return option2
    else:
        raise ValueError("Zoom level not found in the turn.")