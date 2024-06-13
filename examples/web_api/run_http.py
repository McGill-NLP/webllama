from functools import partial
import http.client
import json

import weblinx as wl
import webllama_experimental as wa

def run_http():
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
    action_history_dict = [action.to_dict() for action in action_history]
    state_dict = state.to_dict()

    # Create a connection to the localhost on the port where your server is running
    conn = http.client.HTTPConnection('localhost', 8450)

    # Send a request without parameters to test server response
    conn.request("POST", "/", body=json.dumps({}), headers={'Content-Type': 'application/json'})
    response = conn.getresponse()
    print("Test 1 - Server Initialization Check:")
    print(f"Status: {response.status}")
    print(f"Reason: {response.reason}")
    print(f"Body: {response.read().decode()}\n")
    response.close()

    # Prepare the POST request data
    post_data = json.dumps({
        'action_history': action_history_dict,
        'state': state_dict
    })
    headers = {'Content-Type': 'application/json'}

    # Send a POST request with JSON data
    conn.request("POST", "/", body=post_data, headers=headers)
    response = conn.getresponse()
    print("Test 2 - Functionality Check:")
    print(f"Status: {response.status}")
    print(f"Reason: {response.reason}")
    print(f"Body: {response.read().decode()}")
    response.close()

    # Close the connection
    conn.close()

if __name__ == "__main__":
    run_http()
