import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser(
    prog='acquire.py',
    description='Acquire data from an pool of data for active learning',
)

parser.add_argument('strategy', help = "Acquisition strategy")
parser.add_argument('upool_tsv', help = "Unlabelled pool of data, i.e. all of the data")

parser.add_argument('output_tsv', help = "Path to write acquired data.")
parser.add_argument('--lpool_tsv', help = "Path to data acquired so far (optional)")
parser.add_argument('--seed', help = "Random seed for reproducibility (optional)")

args = parser.parse_args()

# if args.strategy == 'random':

	# Write random sampling function here
	# upool_df = pd.read_tsv(...)
	
	# if args.lpool_tsv
	# 	lpool_df = pd.read_tsv(...)
	#	upool_df = remove from pool to sample the already-used data given in lpool_df

	# new_data = upool_df.sample(frac=.1)

	# run check to make 100% sure new_data and lpool are not overlapping. if so raise an error

	# if check passed, combine lpool and new_data
	# new_data = new_data + lpool_df

	# new_data.to_csv(args.output_tsv, sep="\t")
