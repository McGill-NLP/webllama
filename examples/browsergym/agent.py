from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
import time


from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
import weblinx as wl

import webllama_experimental as wa

from webllama_experimental.integrations.browsergym.functions import (
    say,
    click,
    textinput,
    load,
    scroll,
    wait,
)
from webllama_experimental.integrations.browsergym import replace_bid_with_wl_uid, reverse_dict, postprocess_for_browsergym

def remap_bboxes(bboxes, attrs_map):
    """
    Cleans the bboxes dictionary by replacing the keys with the new unique ids.
    """
    return {attrs_map[k]: v for k, v in bboxes.items()}

class AgentBase(ABC):
    """
    A template class that defines the required signature of an agent interacting with a browsergym environment.
    """

    @abstractmethod
    def reset(self, seed=None) -> None:
        """
        Resets the agent.

        """
        pass

    @abstractmethod
    def get_action(self, obs: dict) -> str:
        """
        Updates the agent with the current observation, and returns its next action (plus an info dict, optional).

        Parameters:
        -----------
        obs: dict
            The current observation of the environment.
        """
        pass

    def preprocess_obs(self, obs: dict) -> dict:
        """Default preprocessing of the observation."""
        pass

    def get_action_mapping(self) -> callable:
        """
        Returns a callable that can be used to map the agent actions to executable python code.
        """
        return None


class WebLinxAgent(AgentBase):
    action_history = None

    def reset(self, seed=None) -> None:
        self.action_history = []
        self.messages = []
        self.start_time = time.time()
        self.has_user_message = False

    @property
    def num_messages(self):
        return len(self.messages)

    @staticmethod
    def get_bboxes(xprops):
        bboxes = {}
        for k in xprops:
            if xprops[k]["visibility"] == 1.0:
                bbox = dict(zip(["x", "y", "width", "height"], xprops[k]["bbox"]))
                # add top, left, bottom, right
                bbox["top"] = bbox["y"]
                bbox["left"] = bbox["x"]
                bbox["bottom"] = bbox["y"] + bbox["height"]
                bbox["right"] = bbox["x"] + bbox["width"]
                bboxes[k] = bbox

        return bboxes

    @staticmethod
    def infer_viewport_from_bboxes(bboxes):
        """
        DO NOT USE THIS, THIS FUNCTION IS NOT WORKING PROPERLY
        """
        if not bboxes:
            return 0, 0

        x = [bboxes[k]["right"] for k in bboxes]
        y = [bboxes[k]["bottom"] for k in bboxes]

        return max(x), max(y)

    def infer_from_screenshot(self, screenshot):
        h, w, _ = screenshot.shape
        return w, h

    @staticmethod
    def get_visible(xprops):
        return {k: xprops[k]["visibility"] == 1.0 for k in xprops}

    @staticmethod
    def rename_uid_attributes(dom_str, new_name="data-webtasks-id", old_name="bid"):
        return dom_str.replace(f"{old_name}=", f"{new_name}=")

    def get_action(self, obs: dict) -> str:
        # preprocessing
        obs["dom_str"] = flatten_dom_to_str(obs["dom_object"])
        obs["bboxes"] = self.get_bboxes(obs["extra_element_properties"])
        # obs["axtree_txt"] = flatten_axtree_to_str(obs["axtree_object"])
        # obs["visible"] = self.get_visible(obs["extra_element_properties"])

        vw, vh = self.infer_from_screenshot(obs["screenshot"])
        obs['html_str_orig'] = self.rename_uid_attributes(obs['dom_str'])

        obs["html_str"], attrs_map = replace_bid_with_wl_uid(obs["dom_str"], return_mapping=True)
        obs["remapped_bboxes"] = remap_bboxes(obs["bboxes"], attrs_map=attrs_map)
        reverse_attrs_map = reverse_dict(attrs_map)

        # check if we have new messages in the chat (+1 will skip first default message)
        new_messages = obs["chat_messages"][self.num_messages + 1 :]
        self.messages.extend(new_messages)

        # update action history with new messages
        for message in new_messages:
            role = "instructor" if message["role"] == "user" else "navigator"
            if role == "instructor":
                self.has_user_message = True

            self.action_history.append(
                wa.classes.Action(
                    type="chat",
                    index=len(self.action_history),
                    intent="say",
                    args={"utterance": message["message"], "speaker": role},
                    timestamp=time.time() - self.start_time,
                    tag=None,
                    attrs=None,
                )
            )
            print(f"New message by '{role}': {message['message']}")

        if not self.has_user_message:
            # sleep and do nothing if no user message has been received
            return "wait(2)"

        state = wa.classes.State(
            index=len(self.action_history),
            html=obs["html_str"],
            bboxes=obs["remapped_bboxes"],
            viewport_height=vh,
            viewport_width=vw,
            type="browser",
        )
        pred_action = wa.web.client.get_prediction(
            self.action_history,
            state,
            address="localhost",
            port=8450,
            max_new_tokens=128,
        )
        # breakpoint()
        pred_action = postprocess_for_browsergym(pred_action, uid_map=reverse_attrs_map)
        # pred_action = postprocess_for_browsergym(pred_action)
        
        a = wa.classes.Action.from_dict(pred_action)

        # add action to action history
        self.action_history.append(a)

        action_str = a.to_str()
        print("Action String:", action_str)

        return action_str

    def get_action_mapping(self) -> callable:
        """
        Returns a callable that can be used to map the agent actions to executable python code.
        """
        action_set = HighLevelActionSet(
            subsets="custom",
            custom_actions=[say, click, textinput, load, scroll, wait],
            multiaction=False,
            strict=True,
        )
        return action_set.to_python_code
