from functools import partial
import time
import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import weblinx as wl
import webllama_experimental as wa

logging.getLogger("urllib3").setLevel(logging.WARNING)

# Prerequisite: We need a `wa.classes.State` object and a list of `wa.classes.Action` objects
# To get that, we will use an example from weblinx, but it's easy to do manually (see below).

demos = wl.list_demonstrations("tests/demonstrations")
replay = wl.Replay.from_demonstration(demos[0])
turn = replay[26]

format_intent_am = partial(
    wa.formatting.build_formatters_action_model(), return_as=dict
)
format_intent_input_dmr, format_intent_out_dmr = wa.formatting.build_formatters_dmr()
action_history = wa.functions.create_action_history_from_replay(
    replay, format_intent_input_dmr, format_intent_am, stop_at_idx=turn.index
)

state = wa.classes.State(
    index=turn.index,
    html=turn.html,
    bboxes=turn.bboxes,
    viewport_height=turn.viewport_height,
    viewport_width=turn.viewport_width,
    type=turn.type,
)


# Verifying that the to_dict and from_dict methods work as expected
act = action_history[0]
d = act.to_dict()
act2 = wa.classes.Action.from_dict(d)
assert act == act2

d = state.to_dict()
state2 = wa.classes.State.from_dict(d)
assert state == state2


# Now, we can start!
# First, load the DMR model we will use to select candidate elements
dmr_name = "McGill-NLP/MiniLM-L6-dmr"
action_model_name = "McGill-NLP/Sheared-LLaMA-2.7B-weblinx"
tokenizer_chat_name = "McGill-NLP/Llama-2-7b-chat-weblinx"

tokenizer = AutoTokenizer.from_pretrained(action_model_name)
tokenizer_chat = AutoTokenizer.from_pretrained(tokenizer_chat_name)
dmr = SentenceTransformer(dmr_name, device="cuda")
action_model = pipeline(model=action_model_name, device=0, torch_dtype="auto")

# We will initialize our processor, which helps us prepare the input for action model
proc = wa.processing.WebTurnProcessor(tokenizer=tokenizer, start_time=time.time())

# Step 1: prepare query, run DMR and prepare retrieved candidates
query_dmr = proc.prepare_dmr_query(action_history, state)
elems = proc.prepare_dmr_elements(state=state)
scores = wa.functions.compute_dmr_scores(dmr, query_dmr, elems)
top_cands, cands_uids = wa.functions.get_top_dmr_candidates(elems, scores, proc.top_k)
cands_str = proc.prepare_candidates(top_cands)

# Step 2: format candidates, utterances, state, and previous actions
html = proc.prepare_state_html(state.html, cands_uids=cands_uids)
utterances = proc.prepare_instructor_chat(action_history, state)
prev_actions = proc.prepare_prev_actions(action_history, state)

# Let's use the default system prompt template, but you can also use your own
sys_prompt_template: str = proc.default_system_prompt_template
sys_prompt = sys_prompt_template.format(
    html=html,
    num_utterances=proc.num_utterances - 1,
    utterances=utterances,
    height=state.viewport_height,
    width=state.viewport_width,
    num_prev_actions=proc.num_prev_actions,
    candidates=cands_str,
)
input_chat = proc.convert_to_chat_list(sys_prompt, prev_actions)

# We can now use the tokenizer's apply_chat_template method to convert it to a format
# that can be used by the action model
input_str = tokenizer_chat.apply_chat_template(input_chat, tokenize=False)

# Let's now pass our input to the action model
output = action_model(
    input_str,
    max_new_tokens=256,
    return_full_text=False,
    batch_size=1,
    pad_token_id=tokenizer.eos_token_id,
)
pred_action = proc.process_action_model_output(
    output=output, index=state.index, elems=elems
)
# optional: For certain platforms you may need to postprocess the action
pred_action = wa.integrations.browsergym.postprocess_for_browsergym(pred_action)
print(pred_action)
# You can now convert this an Action object and add it to the action history
a = wa.classes.Action.from_dict(pred_action)
action_history.append(a)
