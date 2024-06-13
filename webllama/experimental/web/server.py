import abc
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
import logging

from ..classes import Action, State
from ..functions import compute_dmr_scores, get_top_dmr_candidates

# Initialize logging
logging.basicConfig(level=logging.INFO)

class HandlerSimple(BaseHTTPRequestHandler):
    def do_POST(self):
        server = self.server.server_instance  # Accessing the server instance from the HTTPServer
        if not server.model_initialized:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Models not initialized."}).encode('utf-8'))
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # convert json to dict, if it fails we will give a 400 error
        try:
            post_data_dict = json.loads(post_data)
            action_history_dict = post_data_dict.get("action_history", [])
            state_dict = post_data_dict.get("state", {})
            # other kwargs can be passed here
            kwargs = post_data_dict.get("kwargs", {})
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_msg = "Invalid JSON. Please make sure action_history and state are correctly encoded as JSON strings"
            self.wfile.write(json.dumps({"error": error_msg}).encode('utf-8'))
            return
        
        # if state_dict is empty, then we will give an error that reflects that
        if not state_dict:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_msg = "State is empty. Please make sure to provide a valid state."
            self.wfile.write(json.dumps({"error": error_msg}).encode('utf-8'))
            return
    
        response = server.run(action_history_dict, state_dict, **kwargs)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

class ServerBase(abc.ABC):
    model_initialized = False
    dmr = None
    action_model = None
    tokenizer = None
    tokenizer_chat = None

    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        """Initializes models and any necessary components."""
        pass

    @abc.abstractmethod
    def run(self, action_history_json, state_json, **kwargs):
        """Executes the model inference and returns a JSON response."""
        pass

class ServerSimple(ServerBase):
    def initialize(self, dmr_name='McGill-NLP/MiniLM-L6-dmr', action_model_name='McGill-NLP/Llama-3-8B-Web', dmr_device='cuda:0', am_device=0, torch_dtype='auto', save_logs=False):
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        from ..processing import WebTurnProcessor

        logging.info("Initializing models...")
        self.save_logs = save_logs
        self.dmr = SentenceTransformer(dmr_name, device=dmr_device)
        self.action_model = pipeline(model=action_model_name, device=am_device, torch_dtype=torch_dtype)
        self.tokenizer = self.action_model.tokenizer
        self.tokenizer_chat = AutoTokenizer.from_pretrained("McGill-NLP/Llama-3-8B-Web")
        self.proc = WebTurnProcessor(tokenizer=self.action_model.tokenizer, start_time=time.time())
        
        self.model_initialized = True
        self._date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        logging.info("Models initialized.")

    def run(self, action_history_dict, state_dict, **kwargs):

        max_new_tokens = kwargs.get("max_new_tokens", 128)
        
        action_history = [Action.from_dict(d) for d in action_history_dict]
        state = State.from_dict(state_dict)

        # Step 1: prepare query, run DMR and prepare retrieved candidates
        t0 = time.time()
        query_dmr = self.proc.prepare_dmr_query(action_history, state)
        elems = self.proc.prepare_dmr_elements(state=state)
        scores = compute_dmr_scores(self.dmr, query_dmr, elems)
        top_cands, cands_uids = get_top_dmr_candidates(elems, scores, self.proc.top_k)
        cands_str = self.proc.prepare_candidates(top_cands)
        t1 = time.time()
        logging.info(f"Candidate selection completed in {t1 - t0:.2f}s. Running action model...")

        # Step 2: format candidates, utterances, state, and previous actions
        html = self.proc.prepare_state_html(state.html, cands_uids=cands_uids)
        utterances = self.proc.prepare_instructor_chat(action_history, state)
        prev_actions = self.proc.prepare_prev_actions(action_history, state)

        # Let's use the default system prompt template, but you can also use your own
        sys_prompt_template: str = self.proc.default_system_prompt_template
        sys_prompt = sys_prompt_template.format(
            html=html,
            num_utterances=self.proc.num_utterances - 1,
            utterances=utterances,
            height=state.viewport_height,
            width=state.viewport_width,
            num_prev_actions=self.proc.num_prev_actions,
            candidates=cands_str,
        )
        input_chat = self.proc.convert_to_chat_list(sys_prompt, prev_actions)
        input_str = self.tokenizer_chat.apply_chat_template(input_chat, tokenize=False)
        
        # Let's now pass our input to the action model
        output = self.action_model(
            input_str,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            batch_size=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        pred_action = self.proc.process_action_model_output(
            output=output, index=state.index, elems=elems
        )

        t2 = time.time()
        logging.info(f"Action model inference completed in {t2 - t1:.2f}s. Total time: {t2 - t0:.2f}s")

        if self.save_logs:
            # save input_str as a txt file in logs/
            log_dir = Path("logs") / self._date
            log_dir.mkdir(exist_ok=True, parents=True)
            
            with open(log_dir / f"state-{state.index}.txt", "w") as f:
                f.write(input_str)
            
            # save action too as a json file
            with open(log_dir / f"action-{state.index}.json", "w") as f:
                json.dump(pred_action, f, indent=4)
        
        # This is where you'd process your data, omitted for brevity
        return json.dumps(pred_action)

def run_server(server, handler_cls=HandlerSimple, host='localhost', port=8450):
    server_address = (host, port)
    httpd = HTTPServer(server_address, handler_cls)
    httpd.server_instance = server  # Passing the server instance to the HTTPServer
    logging.info(f"Starting server at http://{host}:{port}")
    httpd.serve_forever()

def default_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run a simple server for the LLaMA model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, default="localhost", help="Host address to bind the server to.")
    parser.add_argument("--port", type=int, default=8450, help="Port to bind the server to.")
    parser.add_argument("--dmr_name", type=str, default="McGill-NLP/MiniLM-L6-dmr", help="Name of the DMR model.")
    parser.add_argument("--action_model_name", type=str, default="McGill-NLP/Llama-3-8B-Web", help="Name of the action model on Hugging Face Hub.")
    parser.add_argument("--dmr_device", type=str, default="cuda:0", help="Device to run the DMR model on.")
    parser.add_argument("--am_device", type=int, default=0, help="Device to run the action model on.")
    parser.add_argument("--save_logs", action="store_true", help="Save logs of the server requests.")

    return parser

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    server = ServerSimple()
    server.initialize(
        dmr_name=args.dmr_name,
        action_model_name=args.action_model_name,
        dmr_device=args.dmr_device,
        am_device=args.am_device,
        save_logs=args.save_logs,
    )
    run_server(server, host=args.host, port=args.port)
