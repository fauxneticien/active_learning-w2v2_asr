import pandas as pd
import torch
import torchaudio
import os
import numpy as np
from argparse import ArgumentParser
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ===== parse arguments ===== # 

parser = ArgumentParser(
    prog='acquire.py',
    description='Acquire data from an pool of data for active learning',
)


parser.add_argument('strategy', help = "Acquisition strategy")
parser.add_argument('pool_tsv', help = "Unlabelled pool of data, i.e. all of the data")
parser.add_argument('frac', type = float, help = "Fraction to be sampled from all data")
parser.add_argument('newdata_tsv', help = "Path to write acquired data, i.e. past data + newly sampled (optional)")

parser.add_argument('--lpool_tsv', help = "Path to data acquired so far from past rounds")
parser.add_argument('--seed', help = "Random seed for reproducibility (optional)")
parser.add_argument('--checkpoint', help = "Last checkpoint for model inference")



args = parser.parse_args()
if args.seed:
	np.random.seed(int(args.seed))


# ===== functions to calcualte entropy ===== #

def entropy(logits, dim: int, keepdim: bool = False):
	log_probs = torch.nn.functional.log_softmax(logits, dim=dim, dtype=torch.float32)
	return -torch.sum((torch.exp(log_probs) * log_probs).double(), dim=dim, keepdim=keepdim)


def calculate_entropy(model, processor, wav_path):

	wavform, sample_rate = torchaudio.load(wav_path)
	if sample_rate != 16_000:
		print("Resampling audio to 16 kHz ...")
		samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
		wavform    = samp_to_16k(wavform)# .squeeze().squeeze()
	wavform = wavform.reshape(-1)
	
	input_values = processor(wavform, return_tensors="pt", padding="longest", sampling_rate=16_000).input_values

	with torch.no_grad():
		logits = model(input_values).logits
 
	# take argmax and decode
	predicted_ids = torch.argmax(logits, dim=-1)

	# filter out special tokens
	mask = (predicted_ids > 4) & (predicted_ids != 27)
	indices = torch.nonzero(mask)
	logits_filtered = logits[indices[:,0],indices[:,1],:]
	logits_filtered = torch.unsqueeze(logits_filtered, dim=0)

	experimental_entropy_filtered = entropy(logits_filtered, dim=-1)
	return np.mean(experimental_entropy_filtered.numpy())


# ===== acquire training date ===== #

pool_df = pd.read_csv(args.pool_tsv, sep='\t')
on = ['path', 'text']

if args.lpool_tsv:
	lpool_df = pd.read_csv(args.lpool_tsv, sep='\t')
	upool_df = (pool_df.merge(lpool_df[on], on=on, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', 1))
else:
	new_lpool_df = pd.DataFrame([], columns=on)
	upool_df = pool_df
	lpool_df = new_lpool_df

if args.strategy == 'random':
	# Assign a score to each row randomly
	upool_df = upool_df.assign(score=np.random.rand(upool_df.shape[0]))

if args.strategy == 'entropy':

	# obtain the model
	processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
	model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint)
	if torch.cuda.is_available():
			model.to("cuda")

	# loop through wav files in upool, inference, and get entropy value
	entropy_list = []
	for i in range(upool_df.shape[0]):
		wav_path = upool_df.loc[i].at['path']
		wav_full_path = os.path.join('../data/datasets/gos-kdl', wav_path)
		entropy_result = calculate_entropy(model, processor, wav_full_path)
		entropy_list.append(entropy_result)

	# random for now, change to entropy after
	# upool_df = upool_df.assign(score=np.random.rand(upool_df.shape[0]))
	upool_df = upool_df.assign(score=np.array(entropy_list))



upool_df = upool_df.sort_values(by='score', ascending=False)
new_data_df = upool_df.head(n=int(len(pool_df) * args.frac))

# run check to make 100% sure new_data and lpool are not overlapping. if so raise an error
assert len(lpool_df[lpool_df[on[0]].isin(new_data_df[on[0]])]) == 0, "Newly sampled data cannot overlap with acquired data so far from past."

lpool_combined_df = pd.concat([lpool_df, new_data_df], sort=False)
lpool_combined_df.to_csv(args.newdata_tsv, sep='\t', index=False)
