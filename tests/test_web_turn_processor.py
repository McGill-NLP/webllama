from functools import partial
import time
import logging
import unittest

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import weblinx as wl
import webllama.experimental as wa

logging.getLogger("urllib3").setLevel(logging.WARNING)


class TestWebTurnProcessor(unittest.TestCase):
    def setUp(self):

        demos = wl.list_demonstrations("tests/demonstrations")
        replay = wl.Replay.from_demonstration(demos[0])
        turn = replay[26]

        self.turn = turn
        self.replay = replay
        self.action_model_name = "McGill-NLP/Sheared-LLaMA-2.7B-weblinx"

        self.tokenizer = AutoTokenizer.from_pretrained(self.action_model_name)

        format_intent_input_dmr, format_intent_out_dmr = (
            wa.formatting.build_formatters_dmr()
        )
        format_intent_am = partial(
            wa.formatting.build_formatters_action_model(), return_as=dict
        )
        self.action_history = wa.functions.create_action_history_from_replay(
            replay, format_intent_input_dmr, format_intent_am, stop_at_idx=turn.index
        )
        self.state = wa.classes.State(
            index=turn.index,
            html=turn.html,
            bboxes=turn.bboxes,
            viewport_height=turn.viewport_height,
            viewport_width=turn.viewport_width,
            type=turn.type,
        )

    def test_prepare_dmr_query(self):
        # We will initialize our processor, which helps us prepare the input for action model
        proc = wa.processing.WebTurnProcessor(tokenizer=self.tokenizer)

        # Step 1: prepare query, run DMR and prepare retrieved candidates
        query_dmr = proc.prepare_dmr_query(self.action_history, self.state)

        CORRECT_RESULT = 'Viewport(height=746, width=1536) ---- Instructor Utterances: [00:07] Hello [00:13] Open independent ie Website. [01:30] Go to life and send me some life related news [04:00] Open second one and Summarize the first three paragraphs in a few words ---- Previous Turns:tabswitch(origin=102465633, target=102465635, timestamp="04:19") ; load(url="https://search.yahoo.com/search?fr=mcafee&type=E211US714G0&p=chatgpt", timestamp="04:23") ; click(x=268, y=201, tag="a", attrs={}, timestamp="04:24") ; tabcreate(target=102465636, timestamp="04:25") ; tabswitch(origin=102465635, target=102465636, timestamp="04:25")'
        self.assertIsInstance(query_dmr, str)
        self.assertEqual(query_dmr, CORRECT_RESULT)
