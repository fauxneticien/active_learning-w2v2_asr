import pandas as pd
import os
import numpy as np

from argparse import ArgumentParser

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

args = parser.parse_args()

np.random.seed(int(args.seed))

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

	# random for now, change to entropy after
	upool_df = upool_df.assign(score=np.random.rand(upool_df.shape[0]))

upool_df = upool_df.sort_values(by='score', ascending=False)

new_data_df = upool_df.head(n=int(len(pool_df) * args.frac))

# run check to make 100% sure new_data and lpool are not overlapping. if so raise an error
assert len(lpool_df[lpool_df[on[0]].isin(new_data_df[on[0]])]) == 0, "Newly sampled data cannot overlap with acquired data so far from past."

lpool_combined_df = pd.concat([lpool_df, new_data_df], sort=False)

lpool_combined_df.to_csv(args.newdata_tsv, sep='\t', index=False)
