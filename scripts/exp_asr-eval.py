import os
import pandas as pd
import torchaudio

from datasets import Dataset
from helpers.asr import configure_w2v2_for_inference
from jiwer import wer, cer
import glob

# EVAL_MODELS_DATASETS = [
#     # Evaluation on the same test set using model trained using different amounts of data
#     ("checkpoints/cgn/wav2vec2-large/entropy/1", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-80", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-60", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-40", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-20", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-10", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-05", "data/exps/asr/datasets/test.tsv"),
#     ("data/exps/asr/checkpoints/train-01", "data/exps/asr/datasets/test.tsv")
# ]

EVAL_MODELS_DATASETS = []
TEST_SET_PATH = "data/datasets/cgn/datasets/test.tsv"

parent_dir = "checkpoints/cgn"
model_name_dirs = glob.glob(os.path.join(parent_dir, '*'))
for model_name_dir in model_name_dirs:
    acq_name_dirs = glob.glob(os.path.join(model_name_dir, '*'))
    for acq_name_dir in acq_name_dirs:
        itr_name_dirs = glob.glob(os.path.join(acq_name_dir, '*'))
        for itr_name_dir in itr_name_dirs:
            EVAL_MODELS_DATASETS.append((itr_name_dir, TEST_SET_PATH))

EVAL_RESULTS = []

def read_clip(batch):
    batch['speech'] = torchaudio.load(batch['path'])[0]
    return batch

def make_all_lowercase(batch):
    batch["sentence"] = batch["sentence"].lower()
    batch["transcription"] = batch["transcription"].lower()

    return batch

for model_path, testset_path in EVAL_MODELS_DATASETS:

    print(f"Reading in data from {testset_path} ...")
    test_ds = Dataset.from_pandas(pd.read_csv(testset_path, sep = '\t'))
    test_ds = test_ds.map(read_clip)

    _, processor, transcribe_speech = configure_w2v2_for_inference(model_path)

    print(f"Obtaining predictions using model from {model_path} ...")
    test_ds = test_ds.map(transcribe_speech, remove_columns=["speech"])
    test_ds = test_ds.map(make_all_lowercase)

    EVAL_RESULTS.append({
        "model" : '-'.join(model_path.split('/')[2:]),
        "model_lm" : type(processor).__name__ == 'Wav2Vec2ProcessorWithLM',
        "testset" : os.path.basename(testset_path),
        "wer" : round(wer(test_ds['sentence'], test_ds['transcription']), 3),
        "cer" : round(cer(test_ds['sentence'], test_ds['transcription']), 3)
    })

results_df = pd.DataFrame(EVAL_RESULTS)
results_df.to_csv("test_results.csv", index=False)

print("Results written to test_results")