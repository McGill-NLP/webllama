from functools import partial
import json
import logging
from pathlib import Path

from accelerate import Accelerator
import datasets
from omegaconf import OmegaConf
import hydra
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

import weblinx as wl
from weblinx.processing import load_candidate_elements
from weblinx.processing.prompt import (
    build_input_records_from_selected_turns,
    select_turns_and_candidates_for_prompts,
)
from weblinx.utils.hydra import save_path_to_hydra_logs
from weblinx.utils import set_seed

from .processing import (
    build_formatter_for_multichoice,
    build_prompt_records_for_llama_truncated,
    insert_formatted_chat_into_records,
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    split_path = Path(cfg.data.split_path).expanduser()
    model_save_dir = Path(cfg.model.save_dir).expanduser()
    model_save_dir.mkdir(exist_ok=True, parents=True)
    logging.info(OmegaConf.to_yaml(cfg))

    demo_names = wl.utils.load_demo_names_in_split(split_path, split=cfg.train.split)
    demos = [wl.Demonstration(demo_name, base_dir=cfg.data.base_dir) for demo_name in demo_names]
    candidates = load_candidate_elements(path=cfg.candidates.train_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    model_kwargs['trust_remote_code'] = cfg.model.get('trust_remote_code', False)

    if cfg.train.use_accelerator_device_map:
        accelerator = Accelerator()
        model_kwargs["device_map"] = {"": accelerator.process_index}

    elif cfg.train.use_auto_device_map:
        model_kwargs["device_map"] = "auto"

    if cfg.model.use_flash_attention_2:
        model_kwargs["use_flash_attention_2"] = True
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)

    format_intent = build_formatter_for_multichoice()
    input_records_fname = "input_records_trunc.json"
    build_prompt_records_fn = partial(
        build_prompt_records_for_llama_truncated,
        format_intent=format_intent,
        tokenizer=tokenizer,
    )

    selected_turns = select_turns_and_candidates_for_prompts(
        demos=demos,
        candidates=candidates,
        num_candidates=cfg.candidates.k,
    )

    input_records = build_input_records_from_selected_turns(
        selected_turns=selected_turns,
        format_intent=format_intent,
        build_prompt_records_fn=build_prompt_records_fn,
        format_prompt_records_fn=None,
    )

    template_tokenizer = AutoTokenizer.from_pretrained(cfg.model.template_tokenizer)
    insert_formatted_chat_into_records(
        input_records, template_tokenizer, include_output_target=True
    )

    with open(model_save_dir.joinpath(input_records_fname), "w") as f:
        json.dump(input_records, f, indent=2)

    input_records_texts = [{"text": record["text"]} for record in input_records]

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        optim=cfg.train.optim,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=cfg.train.batch_size_per_device,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.scheduler,
        save_strategy="no",
        evaluation_strategy="no",
        logging_strategy="epoch",
        logging_first_step=True,
        prediction_loss_only=True,
        bf16=True,
        bf16_full_eval=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets.Dataset.from_list(input_records_texts),
        max_seq_length=model.config.max_position_embeddings,
        dataset_text_field="text",
    )

    trainer.train()

    # Save model, tokenizer, trainer state, and path to hydra logs
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    trainer.state.save_to_json(model_save_dir / "trainer_state.json")
    save_path_to_hydra_logs(save_dir=model_save_dir)

    # if the model is saved as pytorch_model_fsdp.bin, rename it to pytorch_model.bin
    fsdp_model_path = model_save_dir / "pytorch_model_fsdp.bin"
    if fsdp_model_path.exists():
        fsdp_model_path.rename(model_save_dir / "pytorch_model.bin")


if __name__ == "__main__":
    main()
