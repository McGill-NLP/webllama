from functools import partial
import webllama_experimental as wa
import weblinx as wl

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

pred_action = wa.web.client.get_prediction(
    action_history, state, address="localhost", port=8450, max_new_tokens=128
)
pred_action = wa.integrations.browsergym.postprocess_for_browsergym(pred_action)
print(pred_action)
a = wa.classes.Action.from_dict(pred_action)
print(a)
