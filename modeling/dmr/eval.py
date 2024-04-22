import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import hydra
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score
import weblinx as wl
from weblinx.processing import group_record_to_dict
from weblinx.utils.recs import ungroup_dict_to_records
from weblinx.utils.hydra import save_path_to_hydra_logs

from .processing import build_records_for_single_demo, build_formatters


def recall_at_k(input_records, k, label_key="label", rank_key="rank"):
    num_correct = 0
    num_total = 0

    for r in input_records:
        if r[label_key] == 1:
            num_total += 1
            if r[rank_key] <= k:
                num_correct += 1

    score = num_correct / num_total
    return score


def mean_reciprocal_rank(input_records, label_key="label", rank_key="rank", k=None):
    if k is None or len(input_records) < k or k < 1:
        k = len(input_records)

    mrr = 0
    num_total = 0

    for r in input_records:
        if r[label_key] == 1:
            if r[rank_key] <= k:
                mrr += 1 / r[rank_key]
            num_total += 1

    mrr /= num_total

    return mrr


def verify_queries_are_all_the_same(grouped_records: dict) -> bool:
    """
    Given a dictionary of grouped records, this function verifies that all
    queries are the same within each group.
    """
    for k, v in grouped_records.items():
        first_query = v[0]["query"]
        if not all(r["query"] == first_query for r in v):
            return False
    return True


def run_model_and_update_groups(
    model, input_grouped: Dict[Any, List[dict]], batch_size, sim_method="cos_sim"
):
    if sim_method == "cos_sim":
        sim_func = cos_sim
    elif sim_method == "dot_product":
        sim_func = dot_score
    else:
        raise ValueError(f"Unknown similarity function: {sim_method}")

    for k, group in tqdm(input_grouped.items(), desc="Computing scores"):
        group = input_grouped[k]
        query = group[0]["query"]
        docs = [r["doc"] for r in group]

        encoded = model.encode(
            [query] + docs, batch_size=batch_size, show_progress_bar=False
        )
        query_vector, doc_vectors = encoded[0], encoded[1:]
        scores = sim_func(query_vector, doc_vectors).cpu().squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]

        for i, r in enumerate(group):
            r["score"] = scores[i]


def build_target_uids_dict(demos, uid_key="data-webtasks-id"):
    """
    Given a list of demonstrations, build a dictionary mapping
    `(demo_name, turn_index) -> uid`. This is used to determine the
    target element for a given demo turn, which labels the element
    as positive or negative.
    """
    target_uids_dict = {}
    for demo in tqdm(demos, desc="Creating dict of target uids"):
        for turn in wl.Replay.from_demonstration(demo):
            if turn.element is None or "attributes" not in turn.element:
                continue
            if uid_key not in turn.element["attributes"]:
                continue

            uid = turn.element["attributes"][uid_key]
            target_uids_dict[(demo.name, turn.index)] = uid

    return target_uids_dict


def get_ranks_from_scores(scores: Dict[Any, float], starts_at=1) -> Dict[Any, int]:
    """
    Given a dictionary of key -> scores, return a dictionary of key -> ranks.
    """
    # Get sorted keys
    keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    ranks = {k: i + starts_at for i, k in enumerate(keys)}

    return ranks


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.seed)

    use_bf16 = cfg.model.use_bf16
    split = cfg.eval.split
    bsize = cfg.eval.batch_size_per_device

    split_path = Path(cfg.data.split_path).expanduser()
    model_save_dir = Path(cfg.model.save_dir).expanduser()
    result_dir = Path(cfg.eval.result_dir).expanduser()

    result_dir.mkdir(parents=True, exist_ok=True)

    if use_bf16:
        torch_dtype = torch.bfloat16
        use_amp = False
    else:
        torch_dtype = torch.float32
        use_amp = True

    # Data loading
    demo_names = wl.utils.load_demo_names_in_split(split_path, split=split)
    demos = [wl.Demonstration(demo_name, base_dir=cfg.data.base_dir) for demo_name in demo_names]

    format_intent_input, _ = build_formatters()
    input_records: List[dict] = []
    logging.info(f"Number of demos: {len(demos)}. Starting building records.")
    for demo in tqdm(demos, desc="Building input records"):
        demo_records = build_records_for_single_demo(
            demo=demo,
            format_intent_input=format_intent_input,
            max_neg_per_turn=None,
            # For eval, we want to include all elements in the demo
            # not just the ones with valid uids
            only_allow_valid_uid=False,
        )
        input_records.extend(demo_records)
    logging.info(f"Completed. Number of input records: {len(input_records)}")

    # Group records by (demo_name, turn_index) pairs
    input_grouped = group_record_to_dict(
        input_records, keys=["demo_name", "turn_index"], remove_keys=False
    )

    # Verify that queries are all the same within each group
    error_msg = "Queries are not all the same within each group"
    assert verify_queries_are_all_the_same(input_grouped), error_msg

    # Run the model and update the scores and ranks in place
    logging.info("Running model and computing scores")

    # Run the model
    model = SentenceTransformer(str(model_save_dir))
    sim_method = cfg.model.get("similarity", "cos_sim")

    logging.info(f"Using the following similarity method: {sim_method}")

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch_dtype):
        run_model_and_update_groups(
            model, input_grouped=input_grouped, batch_size=bsize, sim_method=sim_method
        )
    logging.info("Completed")

    for group in input_grouped.values():
        scores = {r["uid"]: r["score"] for r in group}
        ranks = get_ranks_from_scores(scores)
        for r in group:
            r["rank"] = ranks[r["uid"]]

    # Revert back to original records
    input_records = ungroup_dict_to_records(input_grouped)

    # Metrics
    lengths = np.array([len(v) for v in input_grouped.values()])
    results = {
        "split": split,
        "num_turns": len(input_grouped),
        "num_demos": len(demos),
        "avg_elements_per_turn": lengths.mean(),
        "std_elements_per_turn": lengths.std(),
        "mrr": mean_reciprocal_rank(input_records, k=cfg.eval.mrr_k),
    }

    for k in [1, 5, 10, 20, 50, 100, 200]:
        results[f"recall@{k}"] = recall_at_k(input_records, k=k)

    for k, v in results.items():
        print(f"{k}: {v}")

    # Save results
    with open(result_dir.joinpath("results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save records and scores
    with open(result_dir.joinpath("scores.jsonl"), "w") as f:
        for r in input_records:
            f.write(json.dumps(r) + "\n")

    save_path_to_hydra_logs(save_dir=model_save_dir)


if __name__ == "__main__":
    main()
