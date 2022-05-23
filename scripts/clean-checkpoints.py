import os
import shutil
import glob
import json
import numpy as np
from argparse import ArgumentParser

# ===== parse arguments ===== # 

parser = ArgumentParser(
    prog='clean-checkpoints.py',
    description='Clean checkpoints after fine-tuning',
)


parser.add_argument('parent_dir', help = "checkpoint directory")
args = parser.parse_args()

parent_dir = args.parent_dir

checkpoint_dirs = glob.glob(os.path.join(parent_dir, "checkpoint-*"))
checkpoint_dirs.sort()

last_checkpoint_dir = checkpoint_dirs[-1]

with open(os.path.join(last_checkpoint_dir, "trainer_state.json"), 'r') as f:
    trainer_state = json.loads(f.read())

eval_checkpoints = [ i for i in trainer_state['log_history'] if 'eval_wer' in i ]
eval_wers        = np.array([ i['eval_wer'] for i in eval_checkpoints ])

best_checkpoint     = eval_checkpoints[np.argmin(eval_wers)]
best_checkpoint_dir = os.path.join(parent_dir, 'checkpoint-' + str(best_checkpoint['step']))

for d in checkpoint_dirs:
    if d != best_checkpoint_dir:
        shutil.rmtree(d, ignore_errors=False, onerror=None)
