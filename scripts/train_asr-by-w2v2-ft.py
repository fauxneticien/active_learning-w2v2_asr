import json
import math
import os
import torch

from argparse import ArgumentParser
from datasets import load_metric
import wandb
from helpers.asr import (
    configure_lm,
    configure_w2v2_for_training,
    DataCollatorCTCWithPadding,
    dataset_from_dict,
    get_metrics_computer,
    preprocess_text,
    process_data
)
from transformers import (
    logging,
    Trainer,
    TrainingArguments
)
from helpers.transformers import EarlyStoppingCallback

parser = ArgumentParser(
    prog='train_asr-by-w2v2-ft',
    description='Train an ASR model by fine-tuning a pre-trained wav2vec 2.0 model',
)

parser.add_argument('repo_path_or_name', help = "Pre-trained wav2vec 2.0 model, local path or HuggingFace repo name")
parser.add_argument('output_dir', help = "The output directory where the model predictions and checkpoints will be written")

parser.add_argument('train_tsv', help = "Training data. Two-column tab-separated file with 'path' (path to wav file) and 'sentence' (transcription)")
parser.add_argument('eval_tsv', help = "Evaluation data. Two-column tab-separated file with 'path' (path to wav file) and 'sentence' (transcription)")

parser.add_argument('--use_target_vocab', default=True, help='Use a vocabulary created from target transcriptions (training and evaluation)')

parser.add_argument('--lm_arpa', default=None, help='Path to language model .arpa file (optional)')

parser.add_argument('--hft_logging', default=40, help='HuggingFace Transformers verbosity level (40 = errors, 30 = warnings, 20 = info, 10 = debug)')

args = parser.parse_args()

# Turns out bool('False') evaluates to True in Python (only bool('') is False)
args.use_target_vocab = False if args.use_target_vocab == 'False' else True

logging.set_verbosity(args.hft_logging)

# For debugging
# args.repo_path_or_name = "facebook/wav2vec2-large-robust-ft-swbd-300h"
# args.train_tsv = 'data/train-asr/train.tsv'
# args.eval_tsv  = 'data/train-asr/test.tsv'
# args.output_dir = 'data/asr-temp'
# args.use_target_vocab = False

os.makedirs(args.output_dir, exist_ok=True)

dataset = dataset_from_dict({
    'train': args.train_tsv,
    'eval' : args.eval_tsv
})

w2v2_config = {
    "feature_extractor" : {
        "return_attention_mask" : True
    },
    "model_kwargs" : {
        "mask_time_prob" : 0.075,
        "mask_feature_prob" : 0.008,
        "gradient_checkpointing" : True,
        "ctc_loss_reduction" : "mean"
    }
}

dataset, vocab_dict = preprocess_text(dataset)

model, processor = configure_w2v2_for_training(dataset, args, vocab_dict, w2v2_config)

if args.lm_arpa is not None:
    processor = configure_lm(processor, args.lm_arpa, args.output_dir)

dataset = process_data(dataset, processor)

# Set logging to 'INFO' or else progress bar gets hidden
logging.set_verbosity(20)

# n_epochs   = 200
# batch_size = 32
batch_size = 16

# How many epochs between evals?
# eps_b_eval = 5 
# Save/Eval/Logging steps
# sel_steps = int(math.ceil(len(dataset['train']) / batch_size) * eps_b_eval)

# Learning rate
lr = 1e-5

# set wandb entity and project names
base_model = args.repo_path_or_name.split('/')[-1]
acq_type   = args.output_dir.split('/')[-2]
iteration  = args.train_tsv.split('/')[-1].split('-')[1][0]
run_name   = f"{base_model}-{str(lr)}-{acq_type}-itr{iteration}"

wandb.init(entity="cs224s-project", project="main-exps", name=run_name)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    # gradient_accumulation_steps=1, # when batch_size=32
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    max_steps=2000,
    # fp16=True if torch.cuda.is_available() else False,
    fp16=False, # True could make results worse but computing faster as floats are half precision
    # seed=7135,
    seed=4892,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    learning_rate=lr,
    # Warm up: 500 steps or 10% of total optimisation steps
    # warmup_steps=min(500, int(0.1 * sel_steps * n_epochs)),
    warmup_steps=200,
    # warmup_steps=500,
    # report_to="none",
    # 2022-03-09: manually set optmizier to PyTorch implementation torch.optim.AdamW
    # 'adamw_torch' to get rid of deprecation warning for default optimizer 'adamw_hf'
    optim="adamw_torch",
    metric_for_best_model="wer",
    save_total_limit=4,
    load_best_model_at_end = True,
    # Lower WER is better
    greater_is_better=False,
    dataloader_num_workers=4,
    report_to = 'wandb',
    run_name = run_name
)

trainer = Trainer(
    model=model,
    data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
    args=training_args,
    compute_metrics=get_metrics_computer(processor=processor),
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    tokenizer=processor.feature_extractor,
    callbacks=[ EarlyStoppingCallback(early_stopping_patience=4, min_optim_steps=600) ]
)

print("Training model ...")
trainer.train()
