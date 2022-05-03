import pandas as pd
import os

from argparse import ArgumentParser

parser = ArgumentParser(
    prog='acquire.py',
    description='Acquire data from an pool of data for active learning',
)


parser.add_argument('strategy', help = "Acquisition strategy")
parser.add_argument('pool_tsv', help = "Unlabelled pool of data, i.e. all of the data")
parser.add_argument('lpool_tsv', help = "Path to data acquired so far from past rounds")
parser.add_argument('frac', type = float, help = "Fraction to be sampled from all data")

parser.add_argument('--newdata_tsv', help = "Path to write acquired data, i.e. past data + newly sampled (optional)")
parser.add_argument('--seed', help = "Random seed for reproducibility (optional)")

args = parser.parse_args()

# TODO for later: check if the files already exist
# if args.lpool_tsv == "":
# 	df = pd.DataFrame({'path': [], 'text': []})
# 	df.to_csv('../data/datasets/gos-kdl/datasets/lpool.tsv',index=False)

if args.strategy == 'random':

	pool_df = pd.read_csv(args.pool_tsv, sep='\t')
	lpool_df = pd.read_csv(args.lpool_tsv, sep='\t')

	# candidate = all unpool - acquired so far
	on = ['path', 'text']
	new_data_candidate_df = (pool_df.merge(lpool_df[on], on=on, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', 1))
	new_data_df = new_data_candidate_df.sample(n=int(args.frac*len(pool_df)), replace=False).reset_index(drop=True)

	# run check to make 100% sure new_data and lpool are not overlapping. if so raise an error
	assert len(lpool_df[lpool_df[on[0]].isin(new_data_df[on[0]])]) == 0, "Newly sampled data cannot overlap with acquired data so far from past."

	# if check passed, combine lpool and new_data
	lpool_combined_df = pd.concat([lpool_df, new_data_df])


	# write to files 
	lpool_combined_df.to_csv(args.newdata_tsv, sep='\t', index=False)
	lpool_combined_df.to_csv(args.lpool_tsv, sep='\t', index=False)
