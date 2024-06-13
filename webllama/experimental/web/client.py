from functools import partial
import http.client
import json

from ..classes import State, Action

def get_prediction(action_history, state, address='localhost', port=8450, **kwargs):
    # if action_history is already a list of dictionaries, we can skip this step
    if isinstance(action_history[0], dict):
        action_history_dict = action_history
    elif isinstance(action_history[0], Action):
        action_history_dict = [action.to_dict() for action in action_history]
    else:
        raise ValueError("action_history should be a list of dictionaries or a list of wa.classes.Action objects")
    
    # if state is already a dictionary, we can skip this step
    if isinstance(state, dict):
        state_dict = state
    elif isinstance(state, State):
        state_dict = state.to_dict()
    else:
        raise ValueError("state should be a dictionary or a wa.classes.State object")

    # Create a connection to the localhost on the port where your server is running
    conn = http.client.HTTPConnection(address, port)

    # Prepare the POST request data
    post_data = json.dumps({
        'action_history': action_history_dict,
        'state': state_dict,
        'kwargs': kwargs,
    })

    # Send a POST request with JSON data
    conn.request("POST", "/", body=post_data, headers={'Content-Type': 'application/json'})
    response = conn.getresponse()
    response_body = response.read().decode()
    conn.close()

    # parts response_body into dict
    response_dict = json.loads(response_body)

    return response_dict
