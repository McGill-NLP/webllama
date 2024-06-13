# `webllama.experimental` API

`webllama.experimental` is the new experimental API for working with webllama models. It will eventually be moved to `webllama` directly (once the API is deemed stable).


## Setup

```bash
# Please choose the proper version to ensure you do not break the code
# if there are breaking changes in the future.
# e.g. 0.1.0
pip install webllama=="<version>"
```

You will need to download test demonstrations if you want to run the subsequent scripts that use existing weblinx demonstrations.

```bash
mkdir -p tests/demonstrations
curl -L -o tests/demonstrations/aaabtsd.zip https://github.com/McGill-NLP/weblinx/releases/download/tests-assets/aaabtsd.zip
unzip -u tests/demonstrations/aaabtsd.zip -d tests/demonstrations
curl -L -o tests/demonstrations/aajfwoq.zip https://github.com/McGill-NLP/weblinx/releases/download/tests-assets/aajfwoq.zip
unzip -u tests/demonstrations/aajfwoq.zip -d tests/demonstrations
```

## Quickstart with `webllama.experimental.processing`

To install:
```bash
pip install webllama
# if you want to install transformers, pytorch and sentence-transformers, run:
pip install webllama[modeling]
```

First, you will need to construct your own `action_history` and `state` using `webllama.experimental.classes`:
```python
import webllama.experimental as wa

# Create your action history and state!
action_history = [
    wa.classes.Action(...), # ...
]
state = wa.classes.State(...)
```

You will also need to load your `dmr` and `act_model` models. For example, you can use `transformers` and `sentence-transformers` to load them:
```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

# You can choose your own DMR model, and action model
act_model = pipeline(model=action_model_name, device=0, torch_dtype="auto")
dmr = SentenceTransformer(dmr_name, device="cuda")
```

Now, inside a Python script, you can use the `webllama.experimental.processing` to seamlessly use `Action` and `State` with action model and DMR, and also process the output:

```python
import webllama.experimental as wa

# We will initialize our processor, which helps us prepare the input for action model
proc = wa.processing.WebTurnProcessor(tokenizer=act_model.tokenizer)

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
    utterances=utterances,
    candidates=cands_str,
    # ...
)
input_chat = proc.convert_to_chat_list(sys_prompt, prev_actions)

# Use your tokenizer to convert the input to string and pass it to the action model
input_str = act_model.tokenizer.apply_chat_template(input_chat, tokenize=False)
output = act_model(input_str, ...)
pred_action = proc.process_action_model_output(output, state.index, elems)
a = wa.classes.Action.from_dict(pred_action)
```


## End-to-end example

Here's a full, self-contained example of how to use `webllama.experimental` to interact with a web page using a DMR model and an action model:

```python
from functools import partial
import time
import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import weblinx as wl
import webllama.experimental as wa

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

# Now, we can start!
# First, load the DMR model we will use to select candidate elements
dmr_name = "McGill-NLP/MiniLM-L6-dmr"
action_model_name = "McGill-NLP/Sheared-LLaMA-2.7B-weblinx"
tokenizer_chat_name = "McGill-NLP/Llama-2-7b-chat-weblinx"

tokenizer_chat = AutoTokenizer.from_pretrained(tokenizer_chat_name)
act_model = pipeline(model=action_model_name, device=0, torch_dtype="auto")
dmr = SentenceTransformer(dmr_name, device="cuda")

# We will initialize our processor, which helps us prepare the input for action model
proc = wa.processing.WebTurnProcessor(tokenizer=act_model.tokenizer, start_time=time.time())

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
output = act_model(
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
```

## Tests

To run the tests:

```bash
python -m unittest discover -s tests
```

## Web API

### Running Server

To launch the default server:
```bash
# If you do not want to save logs, omit `--save_logs`
python -m webllama.experimental.web.server --save_logs
```

To create your own server, simply inherit:
```python
from webllama.experimental.web.server import Server

from ..classes import Action, State

# Assuming the classes Action, State, and other necessary imports are already defined
# as provided in your initial setup.

# Initialize logging
logging.basicConfig(level=logging.INFO)

class Server(Server):
    # override initialize and run
    def initialize(self, dmr_name, action_model_name, device, dmr_device, am_device, torch_dtype):
        # initialize your model here

    def run(self, action_history_json, state_json):
        # ...
        pred_action = {
            # ...
        }
        return json.dumps(pred_action)
```

### Connecting via SSH

To connect to the server via SSH, you can use the following command:
```bash
ssh -N -L 8450:localhost:8450 user@server

# Example:
ssh -N -L 8450:localhost:8450 nlp-gpu-2
```

### Using API

You can directly send http request to the web server, or use the client.

Example of HTTP request in python:

```python
from functools import partial
import http.client
import json

from functools import partial
import webllama.experimental as wa
import weblinx as wl

# Prerequisite: We need a `wa.classes.State` object and a list of `wa.classes.Action` objects
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

# Create a connection to the localhost on the port where your server is running
conn = http.client.HTTPConnection('localhost', 8450)

# Prepare the POST request data
post_data = json.dumps({
    'action_history': action_history_dict,
    'state': state_dict
})
headers = {'Content-Type': 'application/json'}

# Send a POST request with JSON data
conn.request("POST", "/", body=post_data, headers=headers)
response = conn.getresponse()
print(f"Status: {response.status}")
print(f"Reason: {response.reason}")
print(f"Body: {response.read().decode()}")
response.close()

# Close the connection
conn.close()
```

### Client

A high level client is provided in `webllama.experimental.web.client`. You can use it as follows:

```python
from functools import partial
import webllama.experimental as wa
import weblinx as wl

# Prerequisite: We need a `wa.classes.State` object and a list of `wa.classes.Action` objects
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

# Now, we can start!
pred_action = wa.web.client.get_prediction(
    action_history, state, address="localhost", port=8450, max_new_tokens=128
)
pred_action = wa.integrations.browsergym.postprocess_for_browsergym(pred_action)
print(pred_action)
a = wa.classes.Action.from_dict(pred_action)
print(a)
```

## Building objects

> Note: This section is a work in progress.

### Build `webllama.experimental.classes.Action`

#### `say` action

```python
utterance_instructor = wa.classes.Action(
    type="chat",
    intent="say",
    index=2,
    args=dict(
        speaker="instructor", utterance="Open independent ie Website.", x=None, y=None
    ),
    timestamp=13.234,
    tag=None,
    attrs=None,
)
```

#### `click` action

To be added.

#### `load` action

To be added.

#### `textinput` action

To be added.

#### `submit` action

To be added.

### Build `webllama.experimental.classes.Bbox`

To be added.

### Build `webllama.experimental.classes.State`

To be added.
