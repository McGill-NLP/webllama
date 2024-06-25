from functools import partial
import time

import lxml.html
import weblinx.utils.format as wlf
import weblinx.processing.prompt as wlpr
from weblinx.processing.dom import clean_and_prune_tree
from weblinx.processing.prompt import (
    find_turns_with_instructor_chat,
    format_candidates,
    format_utterances_truncated,
    multi_attempt_format_prev_turns_truncated,
)
from weblinx.processing.truncation import (
    multi_attempt_truncate_cands_turn,
    multi_attempt_truncate_dom_tree,
)

from .classes import Action, State
from .templates.weblinx import get_system_prompt_template
from .functions import (
    prepare_query_for_dmr,
    prepare_dmr_elements,
    convert_prev_actions_to_chat,
    calculate_remaining_tokens_before_candidates,
    process_action_model_output,
)

class WebTurnProcessor:
    def __init__(
        self,
        tokenizer,
        num_prev_actions=5,
        num_utterances=5,
        top_k=10,
        max_tokens_utterances=5 * 40,
        max_tokens_prev_actions=5 * 50,
        max_tokens_candidates=10 * 65,
        max_html_tokens=700,
        uid_key="data-webtasks-id",
        start_time=None,
    ):
        """
        Initializes the WebTurnProcessor, a class for processing turns in a web-based environment
        like BrowserGym. It is designed to work dynamically rather than with static demonstrations.

        Parameters
        ----------
        tokenizer : callable
            The tokenizer to use for processing.
        num_prev_actions : int, optional
            The number of previous actions to include. Default is 5.
        num_utterances : int, optional
            The number of utterances to include. Default is 5.
        top_k : int, optional
            The number of top candidates to consider. Default is 10.
        max_tokens_utterances : int, optional
            The maximum number of tokens for utterances. Default is 200.
        max_tokens_prev_actions : int, optional
            The maximum number of tokens for previous actions. Default is 250.
        max_tokens_candidates : int, optional
            The maximum number of tokens for candidates. Default is 650.
        max_html_tokens : int, optional
            The maximum number of HTML tokens. Default is 700.
        uid_key : str, optional
            The key for the unique identifier. Default is "data-webtasks-id".
        start_time : float, optional
            The start time of the session. Default is None, which sets it to the current clock time.
        """
        self.tokenizer = tokenizer
        self.num_prev_actions = num_prev_actions
        self.num_utterances = num_utterances
        self.top_k = top_k
        self.max_tokens_utterances = max_tokens_utterances
        self.max_tokens_prev_actions = max_tokens_prev_actions
        self.max_tokens_candidates = max_tokens_candidates
        self.max_html_tokens = max_html_tokens
        self.uid_key = uid_key
        if start_time is None:
            start_time = time.time()
        
        self.start_time = start_time

    @property
    def default_system_prompt_template(self):
        """
        Returns the default system prompt template. It is provided as a property for easy access.

        Returns
        -------
        str
            The default system prompt template used by WebLlama.
        """
        return get_system_prompt_template()

    @staticmethod
    def _format_action_for_dmr(action: Action):
        """
        Formats the action for DMR into a string representing the action.

        Parameters
        ----------
        action : Action
            The action to format into an intent.

        Returns
        -------
        str
            The formatted intent, with the timestamp, attributes, and tag included.
        """
        return action.to_str(
            include_timestamp=True,
            include_attrs=True,
            include_tag=True,
            include_index=False,
            drop_none_coords=True,
            format_timestamp_fn=wlf.format_timestamp,
            ignore_args=("uid",),
        )

    @staticmethod
    def _format_prev_actions_for_action_model(action: Action):
        """
        Formats previous actions for the action model, which is slightly different from DMR.

        Parameters
        ----------
        action : Action
            The action to format.

        Returns
        -------
        dict
            The formatted previous actions.
        """
        return action.to_dict(ignore_args=("x", "y"))

    @staticmethod
    def convert_to_chat_list(sys_prompt, prev_actions):
        """
        Converts to a list of dictionary that can be used by a huggingface tokenizer
        via the apply_chat_template method.

        Parameters
        ----------
        sys_prompt : str
            The system prompt; if you are not sure, use the default system prompt template.
        prev_actions : list of dict
            The previous actions, formatted as a list of dictionaries.

        Returns
        -------
        list of dict
            The converted chat list that can be directly passed to the HF tokenizer.
        """
        return [{"role": "system", "content": sys_prompt}, *prev_actions]

    def prepare_dmr_query(self, action_history, state, return_str=True):
        """
        Prepares a query for the DMR model to use to retrieve candidate elements.

        Parameters
        ----------
        action_history : list of Action
            The history of actions.
        state : State
            The current state.
        return_str : bool, optional
            Whether to return the result as a string. Default is True.

        Returns
        -------
        str or tuple of str
            The prepared DMR query.
        """
        return prepare_query_for_dmr(
            action_history,
            state,
            format_intent=lambda turn, return_as: self._format_action_for_dmr(turn),
            num_prev_turns=self.num_prev_actions,
            num_utterances=self.num_utterances,
            return_str=return_str,
        )

    def prepare_dmr_elements(self, state: State):
        """
        Please refer `webllama.experimental.functions.prepare_dmr_elements` for more details.
        """
        return prepare_dmr_elements(state=state, uid_key=self.uid_key)

    def prepare_instructor_chat(
        self, action_history, state, instructor_name="instructor", format_utterances_fn=wlpr.format_utterances, **kwargs
    ):
        """
        Prepares the instructor chat into a string format that will be used inside the system prompt.

        Parameters
        ----------
        action_history : list of Action
            The history of actions.
        state : State
            The current state.
        instructor_name : str, optional
            The name of the instructor. Default is "instructor".
        format_utterances_fn : callable, optional
            The function to format utterances. Default is wlpr.format_utterances.

        Returns
        -------
        str
            The prepared instructor chat, formatted as a string.
        """
        instructor_chat_turns = find_turns_with_instructor_chat(
            action_history,
            state,
            speaker=instructor_name,
            num_prev_turns=self.num_prev_actions,
        )
        return format_utterances_truncated(
            instructor_chat_turns,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens_utterances,
            num_utterances=self.num_utterances,
            format_utterances_fn=format_utterances_fn,
            **kwargs,
        )

    def prepare_state_html(
        self, state_html, cands_uids, prune_kwargs=None, truncate_kwargs=None
    ):
        """
        Prepares the HTML state into a truncated format that will be formatted as a string
        inside the system prompt.

        Parameters
        ----------
        state_html : str
            The HTML state.
        cands_uids : list of str
            The candidate unique identifiers.
        prune_kwargs : dict, optional
            The keyword arguments for pruning. Default is None.
        truncate_kwargs : dict, optional
            The keyword arguments for truncating. Default is None.

        Returns
        -------
        str
            The prepared HTML state, as a string to be used inside the system prompt.
        """
        if prune_kwargs is None:
            prune_kwargs = {}
        if truncate_kwargs is None:
            truncate_kwargs = {}

        dom_raw = lxml.html.fromstring(state_html)
        dom_pruned = clean_and_prune_tree(
            dom_raw, candidate_uids=cands_uids, **prune_kwargs
        )
        dom_trunc = multi_attempt_truncate_dom_tree(
            dom_tree=dom_pruned,
            tokenizer=self.tokenizer,
            max_tokens=self.max_html_tokens,
            warn_after_attempts=False,
            **truncate_kwargs,
        )
        return dom_trunc["tree_repr"]

    def prepare_prev_actions(
        self, action_history, state, format_intent=None, format_output_dict_fn=None
    ):
        """
        Prepares the previous actions into a truncated format that will be formatted as a string
        inside the system prompt, using the function `convert_prev_actions_to_chat`.

        Parameters
        ----------
        action_history : list of Action
            The history of actions.
        state : State
            The current state.
        format_intent : callable, optional
            The function to format intents. Default is None.
        format_output_dict_fn : callable, optional
            The function to format the output dictionary. Default is None.

        Returns
        -------
        list of dict
            The prepared previous actions. This can be passed to `convert_prev_actions_to_chat`.
        """
        if format_intent is None:
            format_intent = (
                lambda turn, return_as: self._format_prev_actions_for_action_model(turn)
            )

        if format_output_dict_fn is None:
            format_output_dict_fn = partial(
                wlf.format_output_dictionary, function_key="intent"
            )

        prev_actions_trunc = multi_attempt_format_prev_turns_truncated(
            action_history,
            turn=state,
            format_intent=format_intent,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens_prev_actions,
            num_prev_turns=self.num_prev_actions,
            warn_after_attempts=False,
            format_output_dict_fn=format_output_dict_fn,
            turn_sep=None,
        )
        return convert_prev_actions_to_chat(prev_actions_trunc)

    def calculate_remaining_tokens(
        self, html: str, utterances, prev_actions, template_tokenizer=None
    ):
        """
        Calculates the remaining tokens before candidates. This function is not currently
        used by the recommended workflow, but reflects the method presented in the paper to
        maximize the number of tokens input to the action model.

        Parameters
        ----------
        html : str
            The HTML content.
        utterances : str
            The utterances.
        prev_actions : list of dict
            The previous actions.
        template_tokenizer : callable, optional
            The template tokenizer. Default is None.

        Returns
        -------
        int
            The remaining tokens before candidates.
        """
        if template_tokenizer is None:
            template_tokenizer = self.tokenizer

        return calculate_remaining_tokens_before_candidates(
            html=html,
            utterance_context=utterances,
            tokenizer=self.tokenizer,
            tokenizer_chat=template_tokenizer,
            max_html_tokens=self.max_html_tokens,
            max_utterance_tokens=self.max_tokens_utterances,
            max_prev_turns_tokens=self.max_tokens_prev_actions,
            prev_actions_as_chat=prev_actions,
        )

    def prepare_candidates(self, top_cands, format_candidates_fn: callable = None):
        """
        Prepares the candidates into a string that will appear inside the system prompt.

        Parameters
        ----------
        top_cands : list of dict
            The top candidates.
        format_candidates_fn : callable, optional
            The function to format candidates. Default is None.

        Returns
        -------
        str
            The prepared candidates, formatted as a string to be used inside the system prompt.
        """
        if format_candidates_fn is None:
            format_candidates_fn = partial(
                format_candidates, max_char_len=None, use_uid_as_rank=True
            )

        candidates_trunc = multi_attempt_truncate_cands_turn(
            cands_turn=top_cands,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens_candidates,
            format_candidates_fn=format_candidates_fn,
            warn_after_attempts=False,
        )

        cand_str = format_candidates_fn(candidates_trunc, max_char_len=None)

        return cand_str

    def process_action_model_output(self, output, index, elems):
        """
        Please refer `webllama.experimental.functions.process_action_model_output` for more details.
        """
        return process_action_model_output(
            output=output, index=index, elems=elems, start_time=self.start_time
        )